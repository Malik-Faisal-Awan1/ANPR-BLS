export interface Camera {
  id: string;
  name: string;
  location: string;
  status: string;
}

export interface PlateReadEvent {
  id: string;
  cameraId: string;
  timestamp: string;
  plateNumber: string | null;
  confidence: number;
  success: boolean;
  processingMs: number;
  imagePath: string;
}

export interface HistoryResponse {
  data: PlateReadEvent[];
  pagination: {
    total: number;
    page: number;
    limit: number;
    totalPages: number;
  };
}

export interface ConfigData {
  [key: string]: string | number | boolean;
}
