def save_results(cfg, accuracy_matrix, end_time, method_name):
    results_matrix_path = f'{cfg.results_path}/{method_name}_results.txt'
    with open(results_matrix_path, 'w') as f:
        f.write("Accuracy Matrix:\n")
        f.write("Trained / Tested | " + "  | ".join(cfg.tasks) + "\n")
        for i, row in enumerate(accuracy_matrix):
            formatted_row = [f"{score:.2%}" if score is not None else "" for score in row]
            row_string = " | ".join(value for value in formatted_row if value.strip())
            f.write(f"{cfg.tasks[i]}            | " + row_string + "\n")

        f.write(f"\nTotal Execution Time: {end_time:.0f} seconds\n")

    print(f"Results written to {cfg.results_path}")