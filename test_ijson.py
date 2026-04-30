import ijson
import io

f = io.BytesIO(b'[{"a": 1.23}]')
records = list(ijson.items(f, 'item', use_float=True))
print(records)
print(type(records[0]['a']))