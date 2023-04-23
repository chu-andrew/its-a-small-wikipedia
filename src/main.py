import cohere
import wikipedia

import numpy as np
from queue import PriorityQueue
from ratelimit import limits, sleep_and_retry

import cohere_api


def traverse(targets):
    source, destination = targets

    frontier = PriorityQueue()
    explored = {}  # priority_page : parent
    scores = {}  # page : score
    destination_embedding = page_embedding(destination)

    frontier.put((0, source))
    explored[source] = None
    scores[source] = None

    found = False
    while not frontier.empty() and not found:
        _, priority_page = frontier.get()
        print(f"\nOn the Wikipedia page for {priority_page}.")

        if priority_page == destination:
            break

        page_titles = get_links(priority_page)
        embeddings = link_embeddings(page_titles)

        for title, link_embedding in zip(page_titles, embeddings):
            if title not in explored:
                similarity = calculate_similarity(link_embedding, destination_embedding)
                score = 1 / (similarity + 1) * 2

                explored[title] = priority_page
                scores[title] = score

                frontier.put((score, title))

    return explored, scores


def print_path(explored, scores, targets):
    source, destination = targets

    print(f"Path from {source} to {destination} found.")
    path = []
    child = destination
    while child is not source:
        path.append(child)
        child = explored[child]
    path.append(child)
    path = path[::-1]

    for page in path:
        print(page, scores[page])


def page_embedding(word):
    summary = wikipedia.summary(word, auto_suggest=False, redirect=False)
    try:
        embedding = co.embed([summary]).embeddings
        reshaped = np.squeeze(np.asarray(embedding))
        return reshaped
    except cohere.CohereError as e:
        print(e.message)
        print(e.http_status)
        print(e.headers)


@sleep_and_retry
@limits(calls=100, period=60)
def link_embeddings(words):
    try:
        print("Start co:here API call for embeddings.")
        embedding = co.embed(words).embeddings
        print(f"API call returned successfully with {len(embedding)} embeddings.")
        return embedding
    except cohere.CohereError as e:
        print(e.message)
        print(e.http_status)
        print(e.headers)


def calculate_similarity(a, b):
    cosine_similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cosine_similarity if cosine_similarity >= 0 else 0


def standardize_title(title):
    return wikipedia.page(title=title, auto_suggest=False).title


def get_links(page_title):
    try:
        links = wikipedia.page(title=page_title, preload=False, auto_suggest=False).links
        return links
    except wikipedia.exceptions.PageError:
        return Exception


if __name__ == '__main__':
    co = cohere.Client(cohere_api.api_key)
    print("co:here API Connected.")

    source = "Rice University"
    destination = "Moon Jae In"

    source = standardize_title(source)
    destination = standardize_title(destination)
    targets = (source, destination)
    print(f"Find path from {source} to {destination}.")

    explored, scores = traverse(targets)

    print_path(explored, scores, targets)
