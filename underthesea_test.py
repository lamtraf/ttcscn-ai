from underthesea import sent_tokenize


def heuristic_sentence_segmentation(text, max_length=100):
    """
    Segment a Vietnamese text without punctuation into sentences using heuristic rules.

    Parameters:
    text (str): The text to be segmented.
    max_length (int): Maximum character length for a sentence.

    Returns:
    list of str: A list of sentences.
    """
    words = text.split()
    sentences = []
    current_sentence = []

    for word in words:
        current_sentence.append(word)

        # Check if the current sentence has reached the maximum length
        if len(' '.join(current_sentence)) > max_length:
            sentences.append(' '.join(current_sentence))
            current_sentence = []

    # Add the last sentence if it exists
    if current_sentence:
        sentences.append(' '.join(current_sentence))

    return sentences


# Example usage
text = "Đây là một ví dụ Bạn có thể thử với đoạn văn của riêng mình Phân tách câu là một phần quan trọng của xử lý ngôn ngữ tự nhiên"
sentences = heuristic_sentence_segmentation(text, max_length=100)

for sentence in sentences:
    print(sentence)