import sys

def remove_repetitive_text(text):
    def find_earliest_ending_repeating_sequence(s):
        length = len(s)
        earliest_end = float('inf')
        best_seq_end = None

        for i in range(length // 3, length // 2 + 1):  # Checking for different lengths of sequences
            for j in range(length - 2 * i + 1):  # Checking each possible starting position
                seq = s[j:j + i]
                if s[j + i:j + 2 * i] == seq:
                    end_point = j + 2 * i
                    if end_point < earliest_end:
                        earliest_end = end_point
                        best_seq_end = j + i
                        break  # No need to check further sequences at this length once a match is found

        return best_seq_end

    result = find_earliest_ending_repeating_sequence(text)
    if result is not None:
        return text[:result]
    return text

# Example usage
#text1 = "abcdefff..."
#text2 = "9876543543543..."

#print(remove_repetitive_text(text1))  # Output: "abcdef"
#print(remove_repetitive_text(text2))  # Output: "9876543"

# Reading from stdin and applying the function to each line
if __name__ == "__main__":
    for line in sys.stdin:
        cleaned_line = remove_repetitive_text(line.strip())
        print(cleaned_line)
