import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { PneumoService, PredictionResult, AllModelsResult, HealthStatus } from './pneumo.service';

interface ClassInfo {
  label: string;
  description: string;
  urgency: 'low' | 'medium' | 'high';
  urgencyText: string;
  symptoms: string[];
  treatments: string[];
  nextSteps: string[];
  color: string;
  badgeClass: string;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule, HttpClientModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
  // State
  selectedFile: File | null = null;
  previewUrl: string | null = null;
  selectedModel: 'ensemble' | 'resnet' | 'vit' | 'all' = 'ensemble';
  isLoading = false;
  result: PredictionResult | null = null;
  allResults: AllModelsResult | null = null;
  error: string | null = null;
  health: HealthStatus | null = null;
  activeTab: 'result' | 'compare' = 'result';

  readonly classInfo: Record<string, ClassInfo> = {
    Normal: {
      label: 'Normal',
      description: 'No significant pulmonary pathology detected. The lung fields appear clear with no signs of infection, inflammation, or structural abnormality.',
      urgency: 'low',
      urgencyText: 'Routine follow-up recommended',
      symptoms: ['No acute symptoms expected', 'May have mild cough or fatigue', 'Normal respiratory function'],
      treatments: ['No treatment required', 'Maintain healthy lifestyle', 'Regular health check-ups'],
      nextSteps: ['Schedule routine annual check-up', 'Monitor for new symptoms', 'Maintain vaccination records'],
      color: '#00d4aa',
      badgeClass: 'badge-normal'
    },
    Pneumonia: {
      label: 'Pneumonia',
      description: 'Findings consistent with pneumonia — a lung infection causing inflammation and fluid accumulation in the air sacs, impairing gas exchange.',
      urgency: 'medium',
      urgencyText: 'Medical evaluation within 24 hours',
      symptoms: ['High fever and chills', 'Productive cough', 'Chest pain on breathing', 'Shortness of breath', 'Fatigue and weakness'],
      treatments: [
        'Bacterial: Antibiotics (amoxicillin, azithromycin, fluoroquinolones)',
        'Viral: Supportive care; antivirals if influenza confirmed',
        'Rest and adequate hydration',
        'Antipyretics for fever management',
        'Hospitalization if O₂ saturation < 94%'
      ],
      nextSteps: ['Visit GP or emergency care within 24h', 'Blood tests & sputum culture', 'Pulse oximetry monitoring', 'Follow-up X-ray in 6 weeks'],
      color: '#ff6b6b',
      badgeClass: 'badge-pneumonia'
    },
    COVID19: {
      label: 'COVID-19',
      description: 'Bilateral ground-glass opacities and consolidation patterns suggest COVID-19 pneumonia. Peripheral and lower lobe predominance is characteristic.',
      urgency: 'high',
      urgencyText: 'Immediate isolation and PCR testing required',
      symptoms: ['Fever and chills', 'Dry cough', 'Loss of taste/smell', 'Fatigue', 'Shortness of breath', 'Headache and myalgia'],
      treatments: [
        'Mild: Isolation, rest, hydration, antipyretics',
        'Moderate: Antiviral therapy (Paxlovid/nirmatrelvir)',
        'Severe: Dexamethasone, supplemental oxygen',
        'Critical: ICU, mechanical ventilation if needed',
        'Prophylaxis: Vaccination for prevention'
      ],
      nextSteps: ['Isolate immediately', 'PCR/antigen test confirmation', 'Monitor SpO₂ (seek ER if < 94%)', 'Notify close contacts', 'Report to public health if confirmed'],
      color: '#ffbe4d',
      badgeClass: 'badge-covid'
    },
    Tuberculosis: {
      label: 'Tuberculosis',
      description: 'Upper lobe infiltrates and possible cavitation suggest pulmonary tuberculosis. This is a notifiable disease requiring urgent specialist referral.',
      urgency: 'high',
      urgencyText: 'Urgent specialist referral — notifiable disease',
      symptoms: ['Chronic productive cough (> 2 weeks)', 'Night sweats', 'Unexplained weight loss', 'Low-grade fever', 'Haemoptysis (blood in sputum)', 'Fatigue'],
      treatments: [
        'Standard 6-month RIPE therapy:',
        '• Rifampin (R)',
        '• Isoniazid (I)',
        '• Pyrazinamide (P)',
        '• Ethambutol (E)',
        'Drug-resistant TB: Extended specialist regimens',
        'Directly Observed Therapy (DOT) recommended'
      ],
      nextSteps: ['Refer to pulmonologist immediately', 'Sputum AFB smear & culture', 'IGRA or tuberculin skin test', 'Contact tracing', 'Notify public health authorities'],
      color: '#4fa3ff',
      badgeClass: 'badge-tb'
    }
  };

  constructor(private pneumoService: PneumoService) {}

  ngOnInit() {
    this.checkHealth();
  }

  checkHealth() {
    this.pneumoService.checkHealth().subscribe({
      next: (h) => this.health = h,
      error: () => this.health = { status: 'offline', models_loaded: false, error: 'Backend not reachable', device: 'unknown', cuda_available: false }
    });
  }

  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      this.loadFile(input.files[0]);
    }
  }

  onDrop(event: DragEvent) {
    event.preventDefault();
    const file = event.dataTransfer?.files[0];
    if (file && file.type.startsWith('image/')) {
      this.loadFile(file);
    }
  }

  onDragOver(event: DragEvent) { event.preventDefault(); }

  loadFile(file: File) {
    this.selectedFile = file;
    this.result = null;
    this.allResults = null;
    this.error = null;
    const reader = new FileReader();
    reader.onload = (e) => this.previewUrl = e.target?.result as string;
    reader.readAsDataURL(file);
  }

  analyze() {
    if (!this.selectedFile) return;
    this.isLoading = true;
    this.result = null;
    this.allResults = null;
    this.error = null;

    if (this.selectedModel === 'all') {
      this.activeTab = 'compare';
      this.pneumoService.predictAll(this.selectedFile).subscribe({
        next: (res) => { this.allResults = res; this.isLoading = false; },
        error: (err) => { this.error = err.error?.detail || 'Inference failed. Is the backend running?'; this.isLoading = false; }
      });
    } else {
      this.activeTab = 'result';
      const call = this.selectedModel === 'ensemble'
        ? this.pneumoService.predictEnsemble(this.selectedFile)
        : this.selectedModel === 'resnet'
          ? this.pneumoService.predictResnet(this.selectedFile)
          : this.pneumoService.predictVit(this.selectedFile);

      call.subscribe({
        next: (res) => { this.result = res; this.isLoading = false; },
        error: (err) => { this.error = err.error?.detail || 'Inference failed. Is the backend running?'; this.isLoading = false; }
      });
    }
  }

  get currentInfo(): ClassInfo | null {
    const pred = this.result?.prediction;
    return pred ? this.classInfo[pred] : null;
  }

  get sortedProbabilities(): Array<{ key: string; value: number }> {
    if (!this.result?.probabilities) return [];
    return Object.entries(this.result.probabilities)
      .map(([key, value]) => ({ key, value }))
      .sort((a, b) => b.value - a.value);
  }

  barColor(cls: string): string {
    return this.classInfo[cls]?.color || '#888';
  }

  reset() {
    this.selectedFile = null;
    this.previewUrl = null;
    this.result = null;
    this.allResults = null;
    this.error = null;
  }
}
