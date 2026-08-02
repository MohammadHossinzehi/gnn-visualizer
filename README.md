# GNN Visualizer

A real-time WebGL-based graph neural network (GNN) visualizer that enables interactive exploration of GNN computations with GPU-accelerated message passing and force-directed graph layout.

## Features

- **Real-time GNN Visualization**: Watch message passing and node embedding updates as they happen
- **GPU-Accelerated Rendering**: WebGL-based rendering for high-performance visualization of large graphs
- **Force-Directed Layout**: Implements Barnes-Hut quadtree algorithm for O(n log n) force calculations
- **Interactive Controls**: Pan, zoom, and manipulate the graph in real-time
- **Embedding Visualization**: Color-coded nodes based on learned representations
- **Attention Weights**: Display edge attention weights for interpretability
- **Configurable Architecture**: Adjust GNN hyperparameters and layer depths

## Architecture

### GNN Implementation
- Graph Convolutional Network (GCN) layers from scratch in TypeScript
- Backpropagation-through-graph computation for training
- Support for multiple aggregation schemes (mean, sum, max pooling)

### Rendering Pipeline
- Custom WebGL shaders for efficient node and edge rendering
- Separate shaders for different edge types (attention-weighted, structural)
- Dynamic vertex buffer updates during message passing
- Screen-space text rendering for node labels

### Layout Algorithm
- Barnes-Hut approximation of N-body forces
- Octree spatial decomposition for efficient force calculation
- Coulomb repulsion and spring attraction forces
- Iterative refinement with optional GPU offloading

## Getting Started

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

Open http://localhost:8080 in your browser.

### Building for Production

```bash
npm run build
```

## Usage

1. **Load a Graph**: Upload or use the included sample graph (Karate Club or similar)
2. **Configure GNN**: Set number of layers, hidden dimensions, and learning rate
3. **Train**: Click "Start Training" to begin GNN training with visualization
4. **Explore**: Interact with the 3D graph while the model learns
5. **Analyze**: Hover over nodes to see embeddings, click edges to see attention weights

## Technical Details

### Dependencies
- **gl-matrix**: High-performance linear algebra for WebGL transformations
- **webpack**: Module bundling and hot-reload development server
- **TypeScript**: Type-safe implementation

### Performance Characteristics
- Handles graphs with 10k nodes in real-time (60 FPS on modern GPUs)
- Force calculation: O(n log n) via Barnes-Hut
- Rendering: O(n) with constant overhead per edge
- Memory: ~50 bytes per node + 12 bytes per edge + shader overhead

### Browser Support
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+ (with WebGL 2.0)

## Design Decisions

1. **WebGL for Rendering**: Chose WebGL over Canvas/SVG for GPU acceleration—critical for smooth visualization at scale
2. **TypeScript**: Type safety caught numerous matrix operation bugs during development
3. **Barnes-Hut Algorithm**: O(n log n) complexity vs O(n²) essential for graphs with 1000+ nodes
4. **Custom Shaders**: While more code, custom shaders gave fine-grained control over per-node coloring by embedding distance
5. **GPU-Resident Layout**: Position calculations stay on GPU between frames to avoid PCIe overhead

## Testing

Includes unit tests for:
- GCN layer forward/backward passes
- Matrix operations and transformations
- Force calculations against reference implementations
- Shader compilation and WebGL state management

```bash
npm test
```

## License

MIT
