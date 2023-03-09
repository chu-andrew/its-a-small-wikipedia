from collections import deque
import wikipedia

def bfs(start, end):
    frontier = deque([])
    explored = {} # page : parent

    frontier.append(start)
    explored[start] = None

    found = False
    while len(frontier) > 0 and not found:
        page = frontier.popleft()
        print(page, len(frontier))
        
        if page == end: 
            found = True
            break
        
        try: links = wikipedia.page(title=page, auto_suggest=False).links
        except wikipedia.exceptions.PageError: continue

        for linked_page in links:
            if linked_page not in explored:
                explored[linked_page] = page
                frontier.append(linked_page)

                if page == end: # speed up by taking first seen (not true bfs)
                    found = True
                    break

    path = []          
    child = end
    while child is not start:
        path.append(child)
        child = explored[child]
    path.append(child)
    path = path[::-1]

    return path

if __name__=="__main__":

    start = wikipedia.page("breadth first search").title
    end = wikipedia.page("algorithms", auto_suggest=False).title

    print("start: ", start)
    print("end: ", end)

    path = bfs(start, end)
    print(f"Path between {start} and {end}:")
    for page in path: print(page)

