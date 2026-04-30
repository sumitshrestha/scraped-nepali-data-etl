import ijson
import io

f = io.BytesIO(b'[{"nested": {"val": 1.0, "val2": 0.88}}]')
records = list(ijson.items(f, 'item', use_float=True))
print(records)
print(type(records[0]['nested']['val']))
print(type(records[0]['nested']['val2']))