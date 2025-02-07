import time
import streamlit as st
import pandas as pd

# Sorting Algorithms
def bubble_sort(arr, ascending=True):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if (arr[j] > arr[j + 1]) if ascending else (arr[j] < arr[j + 1]):
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                yield arr, f"Bubble Sort: Swapping {arr[j + 1]} and {arr[j]}"

def insertion_sort(arr, ascending=True):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and ((arr[j] > key) if ascending else (arr[j] < key)):
            arr[j + 1] = arr[j]
            j -= 1
            yield arr, f"Insertion Sort: Moving {key} left"
        arr[j + 1] = key
        yield arr, f"Insertion Sort: Placing {key} in position"

def selection_sort(arr, ascending=True):
    n = len(arr)
    for i in range(n):
        idx = i
        for j in range(i + 1, n):
            if (arr[j] < arr[idx]) if ascending else (arr[j] > arr[idx]):
                idx = j
        arr[i], arr[idx] = arr[idx], arr[i]
        yield arr, f"Selection Sort: Swapping {arr[i]} with {arr[idx]}"

def merge_sort(arr, ascending=True):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        yield from merge_sort(left_half, ascending)
        yield from merge_sort(right_half, ascending)

        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if (left_half[i] <= right_half[j]) if ascending else (left_half[i] > right_half[j]):
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1
            yield arr, f"Merge Sort: Merging {arr}"

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1
            yield arr, f"Merge Sort: Adding remaining {arr}"

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1
            yield arr, f"Merge Sort: Adding remaining {arr}"

def quick_sort(arr, ascending=True):
    def quick_sort_helper(arr, low, high):
        if low < high:
            pivot_idx = partition(arr, low, high)
            yield arr, f"Quick Sort: Pivoting at index {pivot_idx}"
            yield from quick_sort_helper(arr, low, pivot_idx - 1)
            yield from quick_sort_helper(arr, pivot_idx + 1, high)

    def partition(arr, low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if (arr[j] <= pivot) if ascending else (arr[j] > pivot):
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    yield from quick_sort_helper(arr, 0, len(arr) - 1)

def heap_sort(arr, ascending=True):
    def heapify(arr, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and ((arr[left] > arr[largest]) if ascending else (arr[left] < arr[largest])):
            largest = left
        if right < n and ((arr[right] > arr[largest]) if ascending else (arr[right] < arr[largest])):
            largest = right

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)

    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
        yield arr, f"Heap Sort: Swapping {arr[i]} with root {arr[0]}"

# Visualization
def visualize_sorting(generator, arr):
    for updated_arr, label in generator:
        # Display sorting step
        st.write(label)
        
        # Update visualization
        st.bar_chart(pd.Series(updated_arr))
        time.sleep(0.2)  # Adjust speed of visualization

# Streamlit App
def main():
    st.title("Sorting Algorithm Visualizer")
    st.write("Lab 2 task by Dr. Urvashi Bansal")

    menu = ["Visualizer", "Exit"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Visualizer":
        n = st.number_input("Enter the number of elements to sort:", min_value=1, max_value=50, step=1)
        arr_input = st.text_input("Enter the elements of the array (space-separated):")

        if st.button("Start Visualization"):
            try:
                arr = list(map(int, arr_input.split()))
                if len(arr) != n:
                    st.error("The number of elements does not match the specified size.")
                else:
                    st.write(f"Original Array: {arr}")

                    algorithm = st.selectbox("Choose a sorting algorithm:", [
                        "Bubble Sort",
                        "Insertion Sort",
                        "Selection Sort",
                        "Merge Sort",
                        "Quick Sort",
                        "Heap Sort",
                    ])

                    ascending = st.radio("Sort Order", ["Ascending", "Descending"]) == "Ascending"

                    # Create a copy of the array to avoid modifying the original array
                    arr_copy = arr.copy()

                    # Get the selected sorting algorithm function
                    algorithms = {
                        "Bubble Sort": bubble_sort,
                        "Insertion Sort": insertion_sort,
                        "Selection Sort": selection_sort,
                        "Merge Sort": merge_sort,
                        "Quick Sort": quick_sort,
                        "Heap Sort": heap_sort,
                    }

                    # Create a new generator for the selected algorithm
                    generator = algorithms[algorithm](arr_copy, ascending)
                    visualize_sorting(generator, arr_copy)

            except ValueError:
                st.error("Please enter valid integers separated by spaces.")

    elif choice == "Exit":
        st.write("Thank you for using the Sorting Algorithm Visualizer!")

if __name__ == "__main__":
    main()
