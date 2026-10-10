def split_large_file(input_file_path, lines_per_file=10000, output_prefix="chunk"):
    """
    Splits a large text file into smaller files.
    Reads line-by-line to handle very large files without running out of memory.
    """
    with open(input_file_path, 'r', encoding='utf-8') as f:
        file_count = 1
        current_lines = []
        
        for line in f:
            current_lines.append(line)
            if len(current_lines) >= lines_per_file:
                output_name = f"{output_prefix}_{file_count}.txt"
                with open(output_name, 'w', encoding='utf-8') as out_f:
                    out_f.writelines(current_lines)
                print(f"Created: {output_name}")
                file_count += 1
                current_lines = []
                
        # Write any remaining lines
        if current_lines:
            output_name = f"{output_prefix}_{file_count}.txt"
            with open(output_name, 'w', encoding='utf-8') as out_f:
                out_f.writelines(current_lines)
            print(f"Created: {output_name}")


def erase_section_by_lines(input_file_path, output_file_path, start_line, end_line):
    """
    Erases content between start_line and end_line (1-indexed, inclusive).
    """
    with open(input_file_path, 'r', encoding='utf-8') as f_in:
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            for current_line_num, line in enumerate(f_in, 1):
                if start_line <= current_line_num <= end_line:
                    continue  # Skip lines in the target range
                f_out.write(line)
    print(f"Content between lines {start_line} and {end_line} removed. Saved to {output_file_path}")


def erase_section_by_markers(input_file_path, output_file_path, start_marker, end_marker):
    """
    Erases content starting from line containing start_marker 
    up to the line containing end_marker.
    """
    erasing = False
    with open(input_file_path, 'r', encoding='utf-8') as f_in:
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                if start_marker in line and not erasing:
                    erasing = True
                    continue  # Skip start line
                
                if end_marker in line and erasing:
                    erasing = False
                    continue  # Skip end line
                
                if not erasing:
                    f_out.write(line)
    print(f"Content between markers removed. Saved to {output_file_path}")


# --- Usage Examples ---

# 1. Split a large file into 50,000-line chunks:
# split_large_file("big_data.txt", lines_per_file=50000, output_prefix="output_part")

# 2. Erase from line 100 to line 500:
erase_section_by_lines("cleaned_data.txt", "cleaned_data_filtered.txt", start_line=724299, end_line=1169749)

# 3. Erase text between custom text flags:
#erase_section_by_markers("H1001_mdp3.txt", "cleaned_data.txt", "=== Output for pb-market-split8-70-4 ===", "Running the learnt policy to solve the queens-30 finally:**************************************************")