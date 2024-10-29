import matplotlib.pyplot as plt


def draw_neural_network(ax, layer_sizes, layer_colors):
    left, right, bottom, top = .1, .9, .1, .9
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    # Draw nodes
    for i, n in enumerate(layer_sizes):
        layer_top = v_spacing * (n - 1) / 2 + (top + bottom) / 2
        color = layer_colors[i % len(layer_colors)]
        for j in range(n):
            circle = plt.Circle((i * h_spacing + left, layer_top - j * v_spacing), v_spacing / 4,
                                color=color, ec='k', zorder=4)
            ax.add_artist(circle)

    # Draw edges
    for i, (n_a, n_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (n_a - 1) / 2 + (top + bottom) / 2
        layer_top_b = v_spacing * (n_b - 1) / 2 + (top + bottom) / 2
        for j in range(n_a):
            for k in range(n_b):
                line = plt.Line2D([i * h_spacing + left, (i + 1) * h_spacing + left],
                                  [layer_top_a - j * v_spacing, layer_top_b - k * v_spacing], c='k')
                ax.add_artist(line)


# Set up plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Define layer sizes and colors
layer_sizes = [4, 5, 6, 3]
layer_colors = ['skyblue', 'lightgreen', 'yellow', '                                                                                                                                                                         ']  # Unique color for each layer

# Draw neural network
draw_neural_network(ax, layer_sizes, layer_colors)

# Save the plot as an image file
plt.savefig("colored_neural_network_diagram.png", format="png", dpi=300)
plt.show()
