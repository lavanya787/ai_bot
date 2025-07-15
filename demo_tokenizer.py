from tokenizer.uml_tokenizer import UnigramTokenizer

# Sample corpus (you can replace this with lines from a file)
corpus = [
    "Electric current is the flow of electric charge.",
    "Ohm's law states that V = IR.",
    "Resistance is measured in ohms.",
    "Current is measured in amperes.",
    "Voltage is the potential difference."
]

# Step 1: Initialize tokenizer
tokenizer = UnigramTokenizer(max_subword_length=10, vocab_size=200)

# Step 2: Train tokenizer on corpus
tokenizer.train(corpus, vocab_size=200, num_iterations=5)

# Step 3: Encode some text
sample_text = "What is electric current?"
encoded = tokenizer.encode(sample_text)
encoded_ids = tokenizer.encode_ids(sample_text)

# Step 4: Decode back to text
decoded_text = tokenizer.decode(encoded)
decoded_from_ids = tokenizer.decode_ids(encoded_ids)

# Step 5: Show token frequencies
frequencies = tokenizer.get_token_frequencies()
sorted_freqs = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)

# Step 6: Display results
print("\n📌 Sample Text:", sample_text)
print("🔢 Encoded Tokens:", encoded)
print("🔢 Encoded IDs:", encoded_ids)
print("🔁 Decoded from Tokens:", decoded_text)
print("🔁 Decoded from IDs:", decoded_from_ids)

print("\n📊 Top 10 Token Frequencies:")
for token, freq in sorted_freqs[:10]:
    print(f"{token:12} ➜ {freq:.2f}")

# Step 7 (Optional): Save and reload
tokenizer.save("demo_tokenizer.json")
tokenizer.load("demo_tokenizer.json")
print("\n💾 Tokenizer saved and reloaded successfully.")
