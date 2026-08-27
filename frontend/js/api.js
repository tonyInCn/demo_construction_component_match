const API_BASE = '';

const Api = {
    async healthCheck() {
        return this.get('/api/health');
    },

    async getModelStatus() {
        return this.get('/api/pipeline/model_status');
    },

    async uploadVideo(file) {
        const formData = new FormData();
        formData.append('file', file);
        return this.post('/api/pipeline/upload_video', formData, false);
    },

    async startDetection(videoPath, device = 'cuda') {
        return this.post('/api/pipeline/start', {
            video_path: videoPath,
            device: device,
        });
    },

    async stopDetection(pipelineId) {
        return this.post(`/api/pipeline/stop/${pipelineId}`);
    },

    async getStatus(pipelineId) {
        return this.get(`/api/pipeline/status/${pipelineId}`);
    },

    async getFrameResults(pipelineId) {
        return this.get(`/api/pipeline/frame_results/${pipelineId}`);
    },

    async getVideoStream(videoPath) {
        return `${API_BASE}/api/pipeline/stream_video?video_path=${encodeURIComponent(videoPath)}`;
    },

    async get(endpoint) {
        const response = await fetch(API_BASE + endpoint);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
    },

    async post(endpoint, data, json = true) {
        const options = {
            method: 'POST',
        };
        if (json) {
            options.headers = { 'Content-Type': 'application/json' };
            options.body = JSON.stringify(data);
        } else {
            options.body = data;
        }
        const response = await fetch(API_BASE + endpoint, options);
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${response.status}`);
        }
        return response.json();
    },
};