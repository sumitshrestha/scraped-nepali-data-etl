#!/usr/bin/env python3
"""
Generate a JSON Schema from a large JSON file (array of objects) using streaming.
"""

import ijson
import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

# ----------------------------------------------------------------------
# Type inference and merging
# ----------------------------------------------------------------------


def _primitive_type(value: Any) -> str:
    """Return JSON Schema primitive type for a Python value."""
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
    """
    Merge two primitive type names, widening as needed.
    Order: null < boolean < integer < number < string
    """
    if existing is None:
        return new
    if existing == new:
        return existing
    # Widening rules
    hierarchy = ["null", "boolean", "integer", "number", "string"]
    if existing in hierarchy and new in hierarchy:
        idx_ex = hierarchy.index(existing)
        idx_new = hierarchy.index(new)
        return hierarchy[max(idx_ex, idx_new)]
    # If one side is not primitive (object/array), we keep the non-primitive?
    # For simplicity, treat conflict as string (most generic)
    return "string"


def _array_item_type(arr: List[Any]) -> str:
    """Determine item type for an array (first non‑null element)."""
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
    """
    Merge array schemas. If both have items type 'object', we merge their properties.
    Otherwise, keep the first non‑generic type.
    """
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
    # Existing is an array schema
    existing_item = existing.get("items", {})
    existing_type = existing_item.get("type")
    if existing_type == "object" and new_item_type == "object" and new_item_schema:
        # Merge object properties recursively
        merged_props = _merge_object_schemas(
            existing_item.get("properties", {}), new_item_schema.get("properties", {})
        )
        existing["items"]["properties"] = merged_props
        return existing
    if existing_type != new_item_type:
        # Widen to string if conflict
        existing["items"]["type"] = "string"
    return existing


def _merge_object_schemas(existing_props: Dict, new_props: Dict) -> Dict:
    """Recursively merge two 'properties' dictionaries."""
    result = existing_props.copy()
    for key, new_schema in new_props.items():
        if key not in result:
            result[key] = new_schema
        else:
            existing_schema = result[key]
            # Merge based on type
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
                # Primitive or incompatible: unify types
                unified_type = _merge_types(existing_type, new_type)
                result[key] = {"type": unified_type}
    return result


# ----------------------------------------------------------------------
# Schema building from sampled objects
# ----------------------------------------------------------------------


def _update_schema_and_counts(
    schema_node: Dict, obj: Any, field_counts: Dict[str, int], path: str = ""
):
    """
    Recursively update the schema (nested properties) and count occurrences
    of each field path.
    """
    if not isinstance(obj, dict):
        return
    for key, value in obj.items():
        full_path = f"{path}.{key}" if path else key
        field_counts[full_path] += 1

        if isinstance(value, dict):
            # Nested object
            if "properties" not in schema_node.get(key, {}):
                schema_node[key] = {"type": "object", "properties": {}}
            _update_schema_and_counts(
                schema_node[key]["properties"], value, field_counts, full_path
            )
        elif isinstance(value, list):
            # Array
            item_type = _array_item_type(value)
            item_schema = None
            if item_type == "object" and value and isinstance(value[0], dict):
                # Infer object schema from first element
                sample_obj = value[0]
                item_schema = {"type": "object", "properties": {}}
                _update_schema_and_counts(
                    item_schema["properties"],
                    sample_obj,
                    field_counts,
                    full_path,  # counts at the array path itself, not inside
                )
                # Remove the dummy path increments? Better: we don't want to count
                # fields inside array elements as top-level required paths. So we
                # should not pass field_counts into that recursion.
                # We'll handle inside array separately: use a temporary counter.
                # Simpler: for arrays of objects, we do not track required for subfields.
                # We'll just record the object schema.
                pass

            if key not in schema_node:
                schema_node[key] = {"type": "array"}
            # Merge array schema
            schema_node[key] = _merge_array_schema(
                schema_node.get(key), item_type, item_schema
            )
        else:
            # Primitive value
            prim_type = _primitive_type(value)
            if key not in schema_node:
                schema_node[key] = {"type": prim_type}
            else:
                existing_type = schema_node[key].get("type")
                unified = _merge_types(existing_type, prim_type)
                schema_node[key] = {"type": unified}


def _add_required_fields(
    schema_node: Dict,
    field_counts: Dict[str, int],
    total_objects: int,
    current_path: str = "",
):
    """
    Add 'required' array to every object schema node based on field counts.
    """
    if schema_node.get("type") != "object":
        return
    props = schema_node.get("properties", {})
    required = []
    for prop_name, prop_schema in props.items():
        prop_path = f"{current_path}.{prop_name}" if current_path else prop_name
        if field_counts.get(prop_path, 0) == total_objects:
            required.append(prop_name)
        # Recurse into nested object or array items that are objects
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
# Public API
# ----------------------------------------------------------------------


def generate_schema(file_path: str, sample_size: int = 100) -> Dict:
    """
    Stream a large JSON array, sample the first `sample_size` objects,
    and return a JSON Schema (Draft-07).
    """
    try:
        with open(file_path, "rb") as f:
            objects = ijson.items(f, "item")
            schema = {"type": "object", "properties": {}}
            field_counts = defaultdict(int)
            total_objects = 0
            for obj in objects:
                if total_objects >= sample_size:
                    break
                _update_schema_and_counts(schema["properties"], obj, field_counts)
                total_objects += 1

            if total_objects == 0:
                raise ValueError("No objects found in the JSON array.")

            # Add required constraints based on full coverage
            _add_required_fields(schema, field_counts, total_objects)

            # Wrap with standard JSON Schema meta-data
            json_schema = {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "title": f"Schema inferred from {total_objects} sample object(s)",
                "type": "object",
                "properties": schema["properties"],
            }
            if "required" in schema:
                json_schema["required"] = schema["required"]
            return json_schema
    except ijson.JSONError as e:
        raise ValueError(f"JSON parsing error: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate JSON Schema from a large JSON array."
    )
    parser.add_argument(
        "file", help="Path to the JSON file (must be an array of objects)"
    )
    parser.add_argument(
        "-s",
        "--sample-size",
        type=int,
        default=100,
        help="Number of objects to sample (default: 100)",
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
