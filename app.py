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
st.title("A* 알고리즘을 이용한 미로 구현")

# 미로 설정
rows = st.slider("미로의 행 수", min_value=5, max_value=20, value=10)
cols = st.slider("미로의 열 수", min_value=5, max_value=20, value=10)

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
    st.write("### 최단 경로 찾기 결과")
    for (x, y) in path:
        maze[x][y] = 0.5  # 경로 표시
    ax.imshow(maze, cmap="viridis")  # 경로를 색상으로 표시
    st.pyplot(fig)
    st.write("최단 경로:", path)
else:
    st.write("경로를 찾을 수 없습니다.")

# Github 배포 안내
st.write("""
### Github와 Streamlit 배포
1. 위 코드를 `.py` 파일로 저장.
2. [Streamlit Community Cloud](https://streamlit.io/)에 접속하여 Github와 연동.
3. 코드를 업로드하고 웹으로 배포.
""")
