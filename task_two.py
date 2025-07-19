import csv
import random
from BTrees.OOBTree import OOBTree
import timeit


def add_to_storage(storage, item):
    price = float(item['Price'])
    if price not in storage:
        storage[price] = []
    storage[price].append(item)

def add_item_to_tree(tree, item):
    add_to_storage(tree, item)


def add_item_to_dict(dct, item):
    add_to_storage(dct, item)


def range_query_tree(tree, min_pirce, max_price):
    results = []
    for price, items in tree.items(min_pirce, max_price):
        results.extend(items)
    return results


def range_query_dict(dct, min_price, max_price):
    results = []
    for price in dct:
        if min_price <= price <= max_price:
            results.extend(dct[price])
    return results


def load_data(file_path):
    tree = OOBTree()
    dct = {}
    prices = []

    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row['Price'] = float(row['Price'])
            prices.append(row['Price'])
            add_item_to_tree(tree, row)
            add_item_to_dict(dct, row)

    return tree, dct, min(prices), max(prices)


def main():
    file_path = 'generated_items_data.csv'
    tree, dct, min_price_data, max_price_data = load_data(file_path)

    num_queries = 100
    range_width = (max_price_data - min_price_data) * 0.1 # всі товари в діапазоні цін, що охоплює 10% від усього цінового діапазону
    queries = []
    for _ in range(num_queries):
        low = random.uniform(min_price_data, max_price_data - range_width)
        high = low + range_width
        queries.append((low, high))

    total_time_tree = 0
    for low, high in queries:
        total_time_tree += timeit.timeit(lambda: range_query_tree(tree, low, high), number=1)

    total_time_dict = 0
    for low, high in queries:
        total_time_dict += timeit.timeit(lambda: range_query_dict(dct, low, high), number=1)
    

    print(f"Total range_query time for OOBTree: {total_time_tree:.6f} seconds")
    print(f"Total range_query time for Dict: {total_time_dict:.6f} seconds")


if __name__ == '__main__':
    main()
