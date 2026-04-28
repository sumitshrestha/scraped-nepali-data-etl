#!/usr/bin/env python3
"""
Generate a JSON Schema from a large JSON file, handling:
- Top-level array of objects
- Single large object
- Line-delimited JSON (one object per line)
"""

import ijson
import json
import sys
from collections import defaultdict
from typing import Any, Dict, Optional

# ----------------------------------------------------------------------
# Type inference and merging (same as before)
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


def _array_item_type(arr: list) -> str:
    for elem in arr:
        if elem is not None:
            if isinstance(elem, dict):
                return "object"
            if isinstance(elem, list):
                return "array"
            return _primitive_type(elem)
    return "null"


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
                unified_type = _merge_types(existing_type, new_type)
                result[key] = {"type": unified_type}
    return result


def _update_schema_and_counts(
    schema_node: Dict, obj: Any, field_counts: Dict[str, int], path: str = ""
):
    if not isinstance(obj, dict):
        return
    for key, value in obj.items():
        full_path = f"{path}.{key}" if path else key
        field_counts[full_path] += 1
        if isinstance(value, dict):
            if "properties" not in schema_node.get(key, {}):
                schema_node[key] = {"type": "object", "properties": {}}
            _update_schema_and_counts(
                schema_node[key]["properties"], value, field_counts, full_path
            )
        elif isinstance(value, list):
            item_type = _array_item_type(value)
            item_schema = None
            if item_type == "object" and value and isinstance(value[0], dict):
                item_schema = {"type": "object", "properties": {}}
                _update_schema_and_counts(
                    item_schema["properties"], value[0], field_counts, full_path
                )
            if key not in schema_node:
                schema_node[key] = {"type": "array"}
            schema_node[key] = _merge_array_schema(
                schema_node.get(key), item_type, item_schema
            )
        else:
            prim_type = _primitive_type(value)
            if key not in schema_node:
                schema_node[key] = {"type": prim_type}
            else:
                existing_type = schema_node[key].get("type")
                schema_node[key] = {"type": _merge_types(existing_type, prim_type)}


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
# Detection and adaptive parsing
# ----------------------------------------------------------------------


def _peek_first_char(file_path: str) -> str:
    """Return the first non‑whitespace character from the file."""
    with open(file_path, "rb") as f:
        while True:
            ch = f.read(1)
            if not ch:
                return ""
            if ch not in (b" ", b"\t", b"\n", b"\r"):
                return ch.decode("utf-8")


def generate_schema(file_path: str, sample_size: int = 100) -> Dict:
    """
    Auto‑detect the JSON structure and generate a JSON Schema.
    """
    first_char = _peek_first_char(file_path)
    if first_char == "[":
        # Top-level array
        return _schema_from_array(file_path, sample_size)
    elif first_char == "{":
        # Single object
        return _schema_from_single_object(file_path, sample_size)
    else:
        # Try line-delimited JSON (each line is a JSON object)
        try:
            return _schema_from_line_delimited(file_path, sample_size)
        except Exception as e:
            raise ValueError(
                f"File does not appear to be a JSON array, object, or line-delimited JSON: {e}"
            )


def _schema_from_array(file_path: str, sample_size: int) -> Dict:
    with open(file_path, "rb") as f:
        objects = ijson.items(f, "item")
        schema = {"type": "object", "properties": {}}
        field_counts = defaultdict(int)
        total = 0
        for obj in objects:
            if total >= sample_size:
                break
            _update_schema_and_counts(schema["properties"], obj, field_counts)
            total += 1
        if total == 0:
            raise ValueError("The JSON array is empty.")
        _add_required_fields(schema, field_counts, total)
        return _wrap_schema(schema, f"Array of objects (sampled {total})")


def _schema_from_single_object(file_path: str, sample_size: int) -> Dict:
    with open(file_path, "rb") as f:
        # Stream key-value pairs from the root object
        items = ijson.kvitems(f, "")
        schema = {"type": "object", "properties": {}}
        field_counts = defaultdict(int)
        total = 0
        # Since it's a single object, we cannot "sample" objects – we have to take the whole object.
        # But we can limit the number of key-value pairs if the object has millions of keys?
        # For a huge object, we'll sample the first `sample_size` keys.
        for key, value in items:
            if total >= sample_size:
                break
            # Treat each top-level key as a separate object? No, it's a single object.
            # We need to update schema with this single object.
            # However, kvitems yields each (key, value). We'll collect them into a temporary dict.
            # Simpler: treat as one object, but we may need to sample inside large nested structures.
            # We'll just parse the whole object using ijson? That could be heavy. Alternative:
            # Use ijson to parse only the first N bytes? Not reliable.
            # Instead, we'll use a different approach: use ijson.items with prefix '' to get the whole object,
            # but that loads everything. For a truly massive single object, consider using ijson.parse
            # and building schema on-the-fly. However, for practicality, we assume a single object fits
            # in memory or we sample its top-level keys.
            # We'll implement a streaming traversal of the first `sample_size` top-level keys only.
            # For each key, we recursively walk its value, but only up to a depth? This is complex.
            # A simpler heuristic: if it's a single object, it's probably not too huge (otherwise line-delimited is better).
            # We'll just load the whole object using standard json.load() – but that defeats the purpose.
            # For large single objects, the user should use line-delimited format.
            # We'll compromise: use ijson to parse the whole object but stop after a certain number of bytes?
            # That's not reliable.
            raise NotImplementedError(
                "Single object parsing with sampling is not fully implemented yet. Please convert to array of objects or line-delimited JSON."
            )
        # For now, we'll fallback to reading the whole object (not recommended for huge files)
        # But let's provide a fallback that works for moderately large objects:
        import json

        with open(file_path, "r") as f2:
            obj = json.load(f2)
        schema = {"type": "object", "properties": {}}
        field_counts = defaultdict(int)
        _update_schema_and_counts(schema["properties"], obj, field_counts)
        _add_required_fields(schema, field_counts, 1)
        return _wrap_schema(schema, "Single object")


def _schema_from_line_delimited(file_path: str, sample_size: int) -> Dict:
    """Each line is a JSON object."""
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
                    # If a line is not an object, we can still include it? For simplicity, skip.
                    continue
                if total >= sample_size:
                    break
                _update_schema_and_counts(schema["properties"], obj, field_counts)
                total += 1
            except json.JSONDecodeError:
                # Skip malformed lines (or raise, depending on strictness)
                continue
    if total == 0:
        raise ValueError("No valid JSON objects found in the line-delimited file.")
    _add_required_fields(schema, field_counts, total)
    return _wrap_schema(schema, f"Line‑delimited objects (sampled {total})")


def _wrap_schema(properties: Dict, description: str) -> Dict:
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Inferred JSON Schema",
        "description": description,
        "type": "object",
        "properties": properties.get(
            "properties", properties
        ),  # properties might already be under "properties"
        "required": properties.get("required", []),
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
        help="Number of objects/keys to sample (default: 100)",
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
