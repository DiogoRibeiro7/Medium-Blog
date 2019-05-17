# Earthmover distance


```
$ python earthmover.py
```

Example usage

```python
p1 = [
    (0, 0),
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
]

p2 = [
    (0, 0),
    (0, 2),
    (0, -2),
    (2, 0),
    (-2, 0),
]

print(earthmover_distance(p1, p2))
```

Example output (with logging):

```
move 0.2 dirt from (0, 0) to (0, 0) for a cost of 0.0
move 0.2 dirt from (0, 1) to (0, 2) for a cost of 0.2
move 0.2 dirt from (0, -1) to (0, -2) for a cost of 0.2
move 0.2 dirt from (1, 0) to (2, 0) for a cost of 0.2
move 0.2 dirt from (-1, 0) to (-2, 0) for a cost of 0.2
0.8
```
