import matplotlib.pyplot as plt

# קישורים בין מפרקים (עצמות)
connections = [
    (0, 1), (1, 20), (20, 2), (2, 3), (3, 4),
    (20, 8), (8, 9), (9, 10),
    (20, 5), (5, 6), (6, 7),
    (0, 16), (0, 12)
]

def parse_frame(frame):
    # הופך מ־150 ערכים => רשימה באורך 50, שכל איבר הוא [x, y, z]
    return [frame[i:i+3] for i in range(0, len(frame), 3)]

def plot_skeleton(flat_frame):
    frame = parse_frame(flat_frame)  # ממיר לפורמט צפוי
    for (i, j) in connections:
        if i >= len(frame) or j >= len(frame):
            continue
        x = [frame[i][0], frame[j][0]]
        y = [frame[i][1], frame[j][1]]
        plt.plot(x, y, 'bo-')
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.pause(0.01)
    plt.clf()
