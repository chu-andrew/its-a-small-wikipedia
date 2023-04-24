import cohere
import wikipedia

import click
import numpy as np
from queue import PriorityQueue
from ratelimit import limits, sleep_and_retry
import time

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

        if priority_page == destination: break

        # retrieve links from wikipedia and embeddings from co:here
        page_titles = wiki_links(priority_page)
        embeddings = link_embeddings(page_titles)

        # calculate similarity scores and update priority queue
        for title, link_embedding in zip(page_titles, embeddings):
            if title not in explored:
                '''
                calculate similarity and score
                similarity range: [0, 1] (higher is more similar, negative similarities clamped to 0)
                need to flip values to work with priority queue
                score range: [0, 1] (lower is more similar)
                '''
                similarity = calculate_similarity(link_embedding, destination_embedding)
                score = 1 - similarity

                # address corner case: ignore disambiguation pages
                if "(disambiguation)" in title: score = 1

                explored[title] = priority_page
                scores[title] = score

                frontier.put((score, title))
                print(title, score)

    return explored, scores


def print_path(explored, scores, targets, duration):
    source, destination = targets

    print(f"Path from {source} to {destination} found in {duration:.2f}s.")

    path = []
    child = destination
    while child is not source:
        path.append(child)
        child = explored[child]
    path.append(child)
    path = path[::-1]

    for page in path:
        print(page, scores[page])


@click.command()
@click.option('--source', prompt="Starting Wikipedia page")
@click.option('--destination', prompt="Destination Wikipedia page")
def cli(source, destination):
    global co
    co = cohere.Client(cohere_api.api_key)
    print("co:here API Connected.")

    source = standardize_title(source)
    destination = standardize_title(destination)
    targets = (source, destination)
    print(f"Find path from {source} to {destination}.")

    start = time.time()
    explored, scores = traverse(targets)
    end = time.time()
    duration = end - start

    print_path(explored, scores, targets, duration)


@sleep_and_retry
@limits(calls=50, period=60)
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


def calculate_similarity(a, b):
    cosine_similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cosine_similarity if cosine_similarity >= 0 else 0  # negative similarities clamped to 0


def wiki_links(page_title):
    try:
        links = wikipedia.page(title=page_title, preload=False, auto_suggest=False).links
        return links
    except wikipedia.exceptions.PageError:
        return Exception


def standardize_title(title):
    return wikipedia.page(title=title, auto_suggest=False).title


if __name__ == "__main__":
    cli()
