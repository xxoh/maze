import streamlit as st
import numpy as np
import heapq
import matplotlib.pyplot as plt

# A* 알고리즘 구현
def astar(maze, start, goal):
    def heuristic(a, b):
        """맨해튼 거리 계산."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    rows, cols = len(maze), len(maze[0])
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and maze[neighbor[0]][neighbor[1]] == 0:
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)

                    if neighbor not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

def reconstruct_path(came_from, current):
    """경로를 역으로 추적하여 반환."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

# Streamlit UI
st.title("A* 알고리즘을 이용한 10x10 미로 최단 경로 찾기")

# 미로 생성 (10x10 고정)
rows, cols = 10, 10
maze = np.random.choice([0, 1], size=(rows, cols), p=[0.7, 0.3])
maze[0][0] = 0  # 시작점
maze[rows-1][cols-1] = 0  # 목표점

# 미로 출력
st.write("### 생성된 미로")
fig, ax = plt.subplots()
ax.imshow(maze, cmap="binary")
ax.set_xticks(np.arange(-0.5, cols, 1))
ax.set_yticks(np.arange(-0.5, rows, 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(color='gray', linestyle='-', linewidth=0.5)

start = (0, 0)
goal = (rows-1, cols-1)

# A* 알고리즘 실행
path = astar(maze, start, goal)

if path:
    # 최단 경로 시각화
    st.write("### 최단 경로 찾기 결과")
    for (x, y) in path:
        maze[x][y] = 0.5  # 경로를 빨간색으로 표시하기 위한 값 설정

    # 경로를 빨간색으로 시각화
    ax.imshow(maze, cmap="coolwarm")  # coolwarm으로 색상 표시 (빨간색 강조)
    for (x, y) in path:
        ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color='red', alpha=0.5))
    
    st.pyplot(fig)

    # 경로 설명
    st.write("### 최단 경로 설명")
    st.write(f"출발점: {start}")
    st.write(f"목표점: {goal}")
    st.write(f"최단 경로 길이: {len(path) - 1} 칸 이동")
    st.write("최단 경로:")
    for idx, (x, y) in enumerate(path):
        st.write(f"{idx + 1}. ({x}, {y}) 위치")
else:
    st.write("경로를 찾을 수 없습니다.")

# 한국어로 최단 경로 설명
st.write("""
### 최단 경로란?
- 이 프로그램은 미로에서 **A* 알고리즘**을 사용해 최단 경로를 계산합니다.
- 최단 경로는 **가장 적은 이동 횟수**로 출발점에서 목표점까지 도달할 수 있는 경로입니다.
- 알고리즘은 현재 위치에서 이동 가능한 경로를 탐색하며, 목표에 더 가까운 경로를 우선적으로 탐색합니다.
""")
