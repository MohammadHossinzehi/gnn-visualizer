import { GCNNetwork } from './gnn';
import { WebGLRenderer } from './renderer';
import { ForceDirectedLayout } from './layout';

interface AppConfig {
  canvas: HTMLCanvasElement;
  nodeCount: number;
  hiddenDim: number;
  layers: number;
  learningRate: number;
}

class GNNVisualizer {
  private gnn: GCNNetwork;
  private renderer: WebGLRenderer;
  private layout: ForceDirectedLayout;
  private canvas: HTMLCanvasElement;
  private isTraining: boolean = false;
  private animationFrameId: number | null = null;

  constructor(config: AppConfig) {
    this.canvas = config.canvas;
    
    // Initialize GNN model
    this.gnn = new GCNNetwork({
      inputDim: 64,
      hiddenDim: config.hiddenDim,
      layers: config.layers,
      learningRate: config.learningRate
    });

    // Initialize WebGL renderer
    this.renderer = new WebGLRenderer({
      canvas: this.canvas,
      width: this.canvas.clientWidth,
      height: this.canvas.clientHeight
    });

    // Initialize force-directed layout
    this.layout = new ForceDirectedLayout({
      nodeCount: config.nodeCount,
      width: this.canvas.clientWidth,
      height: this.canvas.clientHeight,
      damping: 0.95
    });

    this.setupEventListeners();
  }

  private setupEventListeners(): void {
    window.addEventListener('resize', () => this.onWindowResize());
    this.canvas.addEventListener('wheel', (e) => this.onMouseWheel(e), false);
    this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
  }

  private onWindowResize(): void {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    this.renderer.resize(width, height);
    this.layout.resize(width, height);
  }

  private onMouseWheel(event: WheelEvent): void {
    event.preventDefault();
    const delta = event.deltaY > 0 ? 0.9 : 1.1;
    this.renderer.zoom(delta);
  }

  private onMouseMove(event: MouseEvent): void {
    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    this.layout.updateMousePosition(x, y);
  }

  public startTraining(): void {
    if (this.isTraining) return;
    this.isTraining = true;
    this.animate();
  }

  public stopTraining(): void {
    this.isTraining = false;
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
    }
  }

  private animate(): void {
    if (!this.isTraining) return;

    // Update GNN embeddings
    const embeddings = this.gnn.forward();
    
    // Update layout based on forces
    this.layout.update();
    
    // Render frame
    this.renderer.render(
      this.layout.getPositions(),
      embeddings,
      this.layout.getEdges()
    );

    this.animationFrameId = requestAnimationFrame(() => this.animate());
  }

  public getGNN(): GCNNetwork {
    return this.gnn;
  }

  public getRenderer(): WebGLRenderer {
    return this.renderer;
  }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById('canvas') as HTMLCanvasElement;
  
  if (!canvas) {
    console.error('Canvas element not found');
    return;
  }

  const visualizer = new GNNVisualizer({
    canvas,
    nodeCount: 1024,
    hiddenDim: 32,
    layers: 2,
    learningRate: 0.01
  });

  // Expose to window for debugging
  (window as any).visualizer = visualizer;

  // Setup UI controls
  const startBtn = document.getElementById('start-training') as HTMLButtonElement;
  const stopBtn = document.getElementById('stop-training') as HTMLButtonElement;

  if (startBtn) {
    startBtn.addEventListener('click', () => visualizer.startTraining());
  }

  if (stopBtn) {
    stopBtn.addEventListener('click', () => visualizer.stopTraining());
  }
});

export { GNNVisualizer };
