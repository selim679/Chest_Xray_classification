import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface PredictionResult {
  model: string;
  prediction: string;
  confidence: number;
  probabilities: { [key: string]: number };
  inference_ms: number;
  component_predictions?: {
    resnet: { [key: string]: number };
    vit: { [key: string]: number };
  };
  ensemble_weights?: { resnet: number; vit: number };
}

export interface AllModelsResult {
  resnet: PredictionResult;
  vit: PredictionResult;
  ensemble: PredictionResult;
  total_inference_ms: number;
}

export interface HealthStatus {
  status: string;
  models_loaded: boolean;
  error: string | null;
  device: string;
  cuda_available: boolean;
}

@Injectable({ providedIn: 'root' })
export class PneumoService {
  private apiUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  checkHealth(): Observable<HealthStatus> {
    return this.http.get<HealthStatus>(`${this.apiUrl}/health`);
  }

  predictEnsemble(file: File): Observable<PredictionResult> {
    const form = new FormData();
    form.append('file', file);
    return this.http.post<PredictionResult>(`${this.apiUrl}/predict/ensemble`, form);
  }

  predictResnet(file: File): Observable<PredictionResult> {
    const form = new FormData();
    form.append('file', file);
    return this.http.post<PredictionResult>(`${this.apiUrl}/predict/resnet`, form);
  }

  predictVit(file: File): Observable<PredictionResult> {
    const form = new FormData();
    form.append('file', file);
    return this.http.post<PredictionResult>(`${this.apiUrl}/predict/vit`, form);
  }

  predictAll(file: File): Observable<AllModelsResult> {
    const form = new FormData();
    form.append('file', file);
    return this.http.post<AllModelsResult>(`${this.apiUrl}/predict/all`, form);
  }
}
