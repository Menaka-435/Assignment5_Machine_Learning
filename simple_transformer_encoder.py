import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """
    Computes Scaled Dot-Product Attention.
    Q, K, V are numpy arrays of shape (seq_len, d_k)
    """
    
    # Step 1: Compute dot-product scores
    scores = np.matmul(Q, K.T)
    
    # Step 2: Scale scores by sqrt(d_k)
    d_k = Q.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)
    
    # Step 3: Apply softmax to get attention weights
    attention_weights = softmax(scaled_scores)
    
    # Step 4: Compute context vector
    context_vector = np.matmul(attention_weights, V)
    
    return attention_weights, context_vector


# Example run
Q = np.random.rand(3, 4)
K = np.random.rand(3, 4)
V = np.random.rand(3, 4)

weights, context = scaled_dot_product_attention(Q, K, V)

print("Attention Weights:\n", weights)
print("Context Vector:\n", context)
