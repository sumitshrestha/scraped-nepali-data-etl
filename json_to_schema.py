#!/usr/bin/env python3
"""
Generate JSON Schema from large JSON files of any structure:
- Array of objects
- Single huge object
- Line-delimited JSON
"""

import ijson
import json
import sys
from collections import defaultdict
from typing import Dict, Optional, Any

# ----------------------------------------------------------------------
# Type inference and merging
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
        else:
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
# Streaming for a single huge object using ijson.parse
# ----------------------------------------------------------------------


def _stream_schema_from_single_object(file_path: str, max_samples: int = 10000) -> Dict:
    """
    Traverse a huge JSON object using ijson.parse, sampling at most `max_samples`
    primitive values to build the schema incrementally.
    """
    schema = {"type": "object", "properties": {}}
    # We need a map from path to current type and a count of occurrences
    # Instead of full counts, we just need to know if a field appears in all
    # sampled objects – but here we have only *one* logical object,
    # so "required" will include all fields that appear up to the sample limit.
    # For simplicity, we treat each unique path as required if it appears.
    field_occurrences = defaultdict(int)

    with open(file_path, "rb") as f:
        parser = ijson.parse(f)
        # Stack to track current path segments (list of keys/indexes)
        path_stack = []
        # For arrays, we need to know the current array element index (not used for schema)
        # We'll build a dictionary representation of the schema incrementally.
        # We'll maintain a reference to the current schema node as we go deep.
        # This is complex; instead we'll collect field paths and types,
        # then construct the schema after sampling.
        # Simpler: record every field path we encounter and its type,
        # then rebuild the nested object structure later.
        sample_count = 0
        field_types = {}  # path -> merged type
        for prefix, event, value in parser:
            if sample_count >= max_samples:
                break
            if event == "map_key":
                path_stack.append(value)  # push key
            elif event == "start_map":
                pass  # nothing extra
            elif event == "end_map":
                if path_stack:
                    path_stack.pop()
            elif event == "start_array":
                path_stack.append("[]")  # mark array position
            elif event == "end_array":
                if path_stack and path_stack[-1] == "[]":
                    path_stack.pop()
            elif event in ("number", "string", "boolean", "null"):
                # primitive value
                full_path = ".".join(str(p) for p in path_stack if p != "[]")
                if full_path:
                    # For simplicity, we ignore array element indexing
                    # and treat the whole array as a generic array.
                    # We'll handle arrays separately: record that this path is an array,
                    # and the primitive type is the element type.
                    # But events for array elements: the path includes '[]' as a placeholder.
                    # We can detect if the last segment is '[]' -> this is an array element.
                    if path_stack and path_stack[-1] == "[]":
                        # This is an array element. We need to store the array path
                        # and the element type.
                        array_path = ".".join(
                            str(p) for p in path_stack[:-1] if p != "[]"
                        )
                        elem_type = _primitive_type(value)
                        # For arrays, we treat the schema as array with items type
                        # We'll store a special marker
                        field_types[f"{array_path}[]"] = elem_type
                    else:
                        # Regular object field
                        prim_type = _primitive_type(value)
                        if full_path in field_types:
                            field_types[full_path] = _merge_types(
                                field_types[full_path], prim_type
                            )
                        else:
                            field_types[full_path] = prim_type
                sample_count += 1
            elif event == "start_array":
                # Already handled above for path
                continue

    # Convert collected flat field types into nested JSON Schema
    schema_props = {}
    for path, typ in field_types.items():
        if path.endswith("[]"):
            # Array field
            array_path = path[:-2]
            parts = array_path.split(".") if array_path else []
            current = schema_props
            for i, part in enumerate(parts):
                if part not in current:
                    if i == len(parts) - 1:
                        # This is where the array lives
                        current[part] = {"type": "array", "items": {"type": typ}}
                    else:
                        current[part] = {"type": "object", "properties": {}}
                current = (
                    current[part].get("properties", current[part])
                    if i < len(parts) - 1
                    else current[part]
                )
            # For the array, ensure items type
            if parts:
                last = parts[-1]
                current[last]["items"]["type"] = typ
        else:
            # Regular field
            parts = path.split(".")
            current = schema_props
            for i, part in enumerate(parts):
                if part not in current:
                    if i == len(parts) - 1:
                        current[part] = {"type": typ}
                    else:
                        current[part] = {"type": "object", "properties": {}}
                if isinstance(current[part], dict) and i < len(parts) - 1:
                    current = current[part].setdefault("properties", {})
                elif i < len(parts) - 1:
                    # Should be object with properties
                    if "properties" not in current[part]:
                        current[part]["properties"] = {}
                    current = current[part]["properties"]
                else:
                    # leaf
                    pass

    # Mark all fields as required (since a single object, any field that appears is required)
    def mark_required(node):
        if node.get("type") == "object" and "properties" in node:
            node["required"] = list(node["properties"].keys())
            for subnode in node["properties"].values():
                mark_required(subnode)
        elif (
            node.get("type") == "array"
            and "items" in node
            and node["items"].get("type") == "object"
        ):
            mark_required(node["items"])

    mark_required({"type": "object", "properties": schema_props})

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Schema inferred from single huge object (streaming)",
        "type": "object",
        "properties": schema_props,
        "required": list(schema_props.keys()),
    }


def _schema_from_single_object_streaming(
    file_path: str, sample_size: int = 100
) -> Dict:
    """
    Public entry point for a single huge JSON object.
    Uses streaming ijson.parse with limited sample size.
    """
    # sample_size here means number of primitive values to sample
    return _stream_schema_from_single_object(file_path, max_samples=sample_size)


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
        # Top-level array
        return _schema_from_array(file_path, sample_size)
    elif first_char == "{":
        # Single huge object – use streaming parser
        return _schema_from_single_object_streaming(file_path, sample_size)
    else:
        # Try line-delimited JSON
        return _schema_from_line_delimited(file_path, sample_size)


# ----------------------------------------------------------------------
# Implementations for array and line-delimited (same as before, with minor fixes)
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
        help="Number of objects/primitive values to sample (default: 100)",
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
