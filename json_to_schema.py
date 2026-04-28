#!/usr/bin/env python3
"""
Generate JSON Schema from large JSON files:
- Array of objects (streaming)
- Single huge object (streaming, no memory blow‑up)
- Line‑delimited JSON (one object per line)
"""

import ijson
import json
import sys
from collections import defaultdict
from typing import Any, Dict, Optional, List, Tuple

# ----------------------------------------------------------------------
# Type inference and merging (shared)
# ----------------------------------------------------------------------


def _primitive_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    raise ValueError(f"Unsupported type: {type(value)}")


def _merge_types(existing: Optional[str], new: str) -> str:
    if existing is None:
        return new
    if existing == new:
        return existing
    hierarchy = ["null", "boolean", "integer", "number", "string"]
    if existing in hierarchy and new in hierarchy:
        idx_ex = hierarchy.index(existing)
        idx_new = hierarchy.index(new)
        return hierarchy[max(idx_ex, idx_new)]
    return "string"


def _merge_array_schema(
    existing: Optional[Dict], new_item_type: str, new_item_schema: Optional[Dict] = None
) -> Dict:
    if existing is None:
        if new_item_type == "object" and new_item_schema:
            return {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": new_item_schema.get("properties", {}),
                },
            }
        return {"type": "array", "items": {"type": new_item_type}}
    existing_item = existing.get("items", {})
    existing_type = existing_item.get("type")
    if existing_type == "object" and new_item_type == "object" and new_item_schema:
        merged_props = _merge_object_schemas(
            existing_item.get("properties", {}), new_item_schema.get("properties", {})
        )
        existing["items"]["properties"] = merged_props
        return existing
    if existing_type != new_item_type:
        existing["items"]["type"] = "string"
    return existing


def _merge_object_schemas(existing_props: Dict, new_props: Dict) -> Dict:
    result = existing_props.copy()
    for key, new_schema in new_props.items():
        if key not in result:
            result[key] = new_schema
        else:
            existing_schema = result[key]
            existing_type = existing_schema.get("type")
            new_type = new_schema.get("type")
            if existing_type == "object" and new_type == "object":
                existing_schema["properties"] = _merge_object_schemas(
                    existing_schema.get("properties", {}),
                    new_schema.get("properties", {}),
                )
            elif existing_type == "array" and new_type == "array":
                result[key] = _merge_array_schema(
                    existing_schema, new_schema["items"]["type"], new_schema["items"]
                )
            else:
                result[key] = {"type": _merge_types(existing_type, new_type)}
    return result


def _update_schema_with_object(
    schema_props: Dict, obj: Dict, field_counts: Dict[str, int], path: str = ""
):
    """Update schema and counts from a full object (used for arrays or line‑delimited)."""
    if not isinstance(obj, dict):
        return
    for key, value in obj.items():
        full_path = f"{path}.{key}" if path else key
        field_counts[full_path] += 1
        if isinstance(value, dict):
            if "properties" not in schema_props.get(key, {}):
                schema_props[key] = {"type": "object", "properties": {}}
            _update_schema_with_object(
                schema_props[key]["properties"], value, field_counts, full_path
            )
        elif isinstance(value, list):
            item_type = (
                "object"
                if (value and isinstance(value[0], dict))
                else (
                    "array"
                    if (value and isinstance(value[0], list))
                    else (_primitive_type(value[0]) if value else "null")
                )
            )
            item_schema = None
            if item_type == "object" and value and isinstance(value[0], dict):
                item_schema = {"type": "object", "properties": {}}
                _update_schema_with_object(
                    item_schema["properties"], value[0], field_counts, full_path
                )
            if key not in schema_props:
                schema_props[key] = {"type": "array"}
            schema_props[key] = _merge_array_schema(
                schema_props.get(key), item_type, item_schema
            )
        else:
            prim_type = _primitive_type(value)
            if key not in schema_props:
                schema_props[key] = {"type": prim_type}
            else:
                existing_type = schema_props[key].get("type")
                schema_props[key] = {"type": _merge_types(existing_type, prim_type)}


def _add_required_fields(
    schema_node: Dict,
    field_counts: Dict[str, int],
    total_objects: int,
    current_path: str = "",
):
    if schema_node.get("type") != "object":
        return
    props = schema_node.get("properties", {})
    required = []
    for prop_name, prop_schema in props.items():
        prop_path = f"{current_path}.{prop_name}" if current_path else prop_name
        if field_counts.get(prop_path, 0) == total_objects:
            required.append(prop_name)
        if prop_schema.get("type") == "object":
            _add_required_fields(prop_schema, field_counts, total_objects, prop_path)
        elif (
            prop_schema.get("type") == "array"
            and prop_schema.get("items", {}).get("type") == "object"
        ):
            _add_required_fields(
                prop_schema["items"], field_counts, total_objects, prop_path
            )
    if required:
        schema_node["required"] = required


# ----------------------------------------------------------------------
# Streaming for a single huge object (corrected using stack)
# ----------------------------------------------------------------------


def _stream_schema_from_single_object(file_path: str, max_samples: int = 10000) -> Dict:
    """
    Traverse a huge JSON object using ijson.parse, building the schema
    incrementally with a stack of current schema nodes.
    """
    # Stack of (schema_dict, is_array_element) where schema_dict is the current
    # properties dict (for an object) or the array's item schema (for an array).
    # We'll maintain a separate stack for the current path for merging.
    # Simpler: use a single stack where each entry is a dict representing the
    # current node's "properties" or the array's "items". We'll also keep track
    # of the type of node (object, array) to know how to attach new fields.
    stack = []  # each element: (current_dict, node_type, key_being_built)
    # node_type: 'object' for properties dict, 'array' for items dict
    # For objects, current_dict is the properties dict.
    # For arrays, current_dict is the items schema (e.g., {"type": "..."}).

    # We also need to record field occurrence counts to determine required fields.
    # Since this is a single object, any field that appears is required.
    # So we can simply mark all fields as required at the end.

    # We'll keep a separate structure to count occurrences per full path
    # (same as before) but for merging we rely on the schema building.
    # Let's implement correct incremental merging.

    # Initialize root schema
    root_schema = {"type": "object", "properties": {}}
    # Stack entry: (properties_dict, 'object', None)
    stack.append((root_schema["properties"], "object", None))

    with open(file_path, "rb") as f:
        parser = ijson.parse(f)
        sample_count = 0
        for prefix, event, value in parser:
            if sample_count >= max_samples:
                break

            if event == "map_key":
                # We are inside an object (the top of stack should be an object)
                # The current key is 'value'
                key = value
                # Push a new object entry for this key? No, we will handle when
                # we see start_map or start_array. For now, just store the key
                # temporarily. We'll store it as the "key_being_built" in the stack.
                if stack and stack[-1][1] == "object":
                    # Replace the last entry with the same but with key_being_built set
                    cur_dict, cur_type, _ = stack.pop()
                    stack.append((cur_dict, cur_type, key))
                else:
                    # Should not happen
                    pass

            elif event == "start_map":
                # A new nested object
                if stack and stack[-1][1] == "object":
                    cur_dict, cur_type, current_key = stack[-1]
                    if current_key is not None:
                        # This map is the value for current_key
                        # Create a new properties dict for this nested object
                        if current_key not in cur_dict:
                            cur_dict[current_key] = {"type": "object", "properties": {}}
                        elif cur_dict[current_key].get("type") != "object":
                            # Previous type was a primitive, upgrade to object? This is unlikely.
                            cur_dict[current_key] = {"type": "object", "properties": {}}
                        # Push the new properties dict onto the stack
                        stack.append(
                            (cur_dict[current_key]["properties"], "object", None)
                        )
                    else:
                        # Should not happen: map_key should have set a key
                        pass
                elif stack and stack[-1][1] == "array":
                    # We are inside an array and encountering an object as an element
                    # The stack's top is the array's items schema (should be an object)
                    cur_dict, cur_type, _ = stack[-1]
                    if cur_dict.get("type") != "object":
                        # Need to set items to object with properties
                        cur_dict["type"] = "object"
                        cur_dict["properties"] = {}
                    # Push the array's items properties onto stack
                    stack.append((cur_dict["properties"], "object", None))
                else:
                    # Should not happen
                    pass

            elif event == "end_map":
                # Pop the stack
                if stack:
                    stack.pop()

            elif event == "start_array":
                # A new array
                if stack and stack[-1][1] == "object":
                    cur_dict, cur_type, current_key = stack[-1]
                    if current_key is not None:
                        # This array is the value for current_key
                        if current_key not in cur_dict:
                            cur_dict[current_key] = {"type": "array", "items": {}}
                        elif cur_dict[current_key].get("type") != "array":
                            # Upgrade to array? Not likely.
                            cur_dict[current_key] = {"type": "array", "items": {}}
                        # Push the array's items schema onto stack (initially empty)
                        stack.append((cur_dict[current_key]["items"], "array", None))
                    else:
                        pass
                else:
                    # Should not happen
                    pass

            elif event == "end_array":
                # Pop the stack
                if stack:
                    stack.pop()

            elif event in ("number", "string", "boolean", "null"):
                # Primitive value
                prim_type = _primitive_type(value) if event != "null" else "null"
                if stack and stack[-1][1] == "object":
                    cur_dict, cur_type, current_key = stack[-1]
                    if current_key is not None:
                        # This primitive is the value for current_key
                        if current_key not in cur_dict:
                            cur_dict[current_key] = {"type": prim_type}
                        else:
                            existing_type = cur_dict[current_key].get("type")
                            cur_dict[current_key] = {
                                "type": _merge_types(existing_type, prim_type)
                            }
                        # Clear the key_being_built in the stack
                        stack.pop()
                        stack.append((cur_dict, cur_type, None))
                    else:
                        # Primitive value as an array element? Not tracked separately; we treat array items later.
                        # For arrays, we merge item types only when we see the first element? We'll implement in start_array.
                        pass
                elif stack and stack[-1][1] == "array":
                    # This primitive is an element of the array
                    cur_dict, cur_type, _ = stack[-1]  # cur_dict is the items schema
                    existing_type = cur_dict.get("type")
                    merged = _merge_types(existing_type, prim_type)
                    cur_dict["type"] = merged
                sample_count += 1

    # After traversal, mark all fields as required (since it's a single object)
    def mark_all_required(node):
        if node.get("type") == "object" and "properties" in node:
            node["required"] = list(node["properties"].keys())
            for subnode in node["properties"].values():
                mark_all_required(subnode)
        elif (
            node.get("type") == "array"
            and "items" in node
            and node["items"].get("type") == "object"
        ):
            mark_all_required(node["items"])

    # Apply to root properties
    mark_all_required({"type": "object", "properties": root_schema["properties"]})

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": f"Schema inferred from single huge object (sampled {sample_count} primitive values)",
        "type": "object",
        "properties": root_schema["properties"],
        "required": list(root_schema["properties"].keys()),
    }


# ----------------------------------------------------------------------
# Detection and dispatching
# ----------------------------------------------------------------------


def _peek_first_char(file_path: str) -> str:
    with open(file_path, "rb") as f:
        while True:
            ch = f.read(1)
            if not ch:
                return ""
            if ch not in (b" ", b"\t", b"\n", b"\r"):
                return ch.decode("utf-8")


def generate_schema(file_path: str, sample_size: int = 100) -> Dict:
    first_char = _peek_first_char(file_path)
    if first_char == "[":
        return _schema_from_array(file_path, sample_size)
    elif first_char == "{":
        return _stream_schema_from_single_object(file_path, max_samples=sample_size)
    else:
        return _schema_from_line_delimited(file_path, sample_size)


# ----------------------------------------------------------------------
# Implementations for array and line-delimited
# ----------------------------------------------------------------------


def _schema_from_array(file_path: str, sample_size: int) -> Dict:
    with open(file_path, "rb") as f:
        objects = ijson.items(f, "item")
        schema = {"type": "object", "properties": {}}
        field_counts = defaultdict(int)
        total = 0
        for obj in objects:
            if total >= sample_size:
                break
            _update_schema_with_object(schema["properties"], obj, field_counts)
            total += 1
        if total == 0:
            raise ValueError("The JSON array is empty.")
        _add_required_fields(schema, field_counts, total)
        return _wrap_schema(schema, f"Array of objects (sampled {total})")


def _schema_from_line_delimited(file_path: str, sample_size: int) -> Dict:
    schema = {"type": "object", "properties": {}}
    field_counts = defaultdict(int)
    total = 0
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                if total >= sample_size:
                    break
                _update_schema_with_object(schema["properties"], obj, field_counts)
                total += 1
            except json.JSONDecodeError:
                continue
    if total == 0:
        raise ValueError("No valid JSON objects found in line-delimited file.")
    _add_required_fields(schema, field_counts, total)
    return _wrap_schema(schema, f"Line‑delimited objects (sampled {total})")


def _wrap_schema(schema_node: Dict, description: str) -> Dict:
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Inferred JSON Schema",
        "description": description,
        "type": "object",
        "properties": schema_node.get("properties", schema_node),
        "required": schema_node.get("required", []),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate JSON Schema from any large JSON file (array, single object, or line-delimited)."
    )
    parser.add_argument("file", help="Path to the JSON file")
    parser.add_argument(
        "-s",
        "--sample-size",
        type=int,
        default=100,
        help="Number of primitive values to sample for a single object, "
        "or number of objects for array/line-delimited (default: 100)",
    )
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    args = parser.parse_args()

    try:
        schema = generate_schema(args.file, args.sample_size)
        output = json.dumps(schema, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
        else:
            print(output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
