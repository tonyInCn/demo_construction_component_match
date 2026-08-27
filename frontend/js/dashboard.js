const DETECTION_STEPS = [
    { key: '吊机检测', label: '吊机检测' },
    { key: '确定检测区域', label: '确定检测区域' },
    { key: 'GroundingDINO检测', label: 'GroundingDINO 检测' },
    { key: '传统CV检测', label: '传统CV 检测' },
    { key: 'NMS融合去重', label: 'NMS 融合去重' },
    { key: '多方法构件检测完成', label: '多方法检测完成' },
    { key: '构件匹配与判定', label: '构件匹配与判定' },
];

let state = {
    modelStatus: null,
    pipelineId: null,
    videoPath: null,
    videoFilename: null,
    isDetecting: false,
    statusPoller: null,
    resultsPoller: null,
    completedSteps: new Set(),
    currentStep: null,
};

const els = {
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),
    videoContainer: document.getElementById('videoContainer'),
    video: document.getElementById('video'),
    videoPlaceholder: document.getElementById('videoPlaceholder'),
    fileInput: document.getElementById('fileInput'),
    btnSelect: document.getElementById('btnSelect'),
    btnStart: document.getElementById('btnStart'),
    btnStop: document.getElementById('btnStop'),
    progressBar: document.getElementById('progressBar'),
    progressText: document.getElementById('progressText'),
    progressFrame: document.getElementById('progressFrame'),
    detectionsTable: document.getElementById('detectionsTable'),
    detectionSteps: document.getElementById('detectionSteps'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingText: document.getElementById('loadingText'),
    modelInitialized: document.getElementById('modelInitialized'),
    modelGroundingDino: document.getElementById('modelGroundingDino'),
    modelClip: document.getElementById('modelClip'),
    modelLoadProgress: document.getElementById('modelLoadProgress'),
    uniqueCount: document.getElementById('uniqueCount'),
    currentFps: document.getElementById('currentFps'),
    elapsedTime: document.getElementById('elapsedTime'),
};

async function init() {
    try {
        await Api.healthCheck();
        updateStatus('loading', '加载中...');
        await loadModelStatus();
        startModelStatusPolling();
        renderPipelineSteps();
    } catch (e) {
        updateStatus('error', '连接失败');
        console.error(e);
    }
}

function updateStatus(type, text) {
    els.statusDot.className = 'status-dot';
    if (type === 'ready') {
        els.statusDot.classList.add('ready');
    } else if (type === 'loading') {
        els.statusDot.classList.add('loading');
    } else if (type === 'error') {
        els.statusDot.classList.add('error');
    }
    els.statusText.textContent = text;
}

async function loadModelStatus() {
    try {
        const status = await Api.getModelStatus();
        state.modelStatus = status;
        renderModelStatus(status);

        if (status.initialized) {
            updateStatus('ready', '就绪');
            hideLoading();
        } else if (status.loading) {
            updateStatus('loading', `加载中 ${status.load_progress}%`);
            showLoading(status.load_step);
        } else if (status.load_error) {
            updateStatus('error', '加载失败');
            showLoading(status.load_error, true);
        } else {
            updateStatus('loading', '初始化中...');
            showLoading('初始化中...');
        }
    } catch (e) {
        console.error('加载模型状态失败:', e);
    }
}

function renderModelStatus(status) {
    els.modelInitialized.textContent = status.initialized ? '就绪' : '未就绪';
    els.modelInitialized.className = status.initialized ? 'badge badge-green' : 'badge badge-red';

    els.modelGroundingDino.textContent = status.grounding_dino_loaded ? '已加载' : '未加载';
    els.modelGroundingDino.className = status.grounding_dino_loaded ? 'badge badge-green' : 'badge badge-red';

    els.modelClip.textContent = status.clip_loaded ? '已加载' : '未加载';
    els.modelClip.className = status.clip_loaded ? 'badge badge-green' : 'badge badge-red';

    els.modelLoadProgress.textContent = `${status.load_progress || 0}%`;
}

function renderPipelineSteps() {
    els.detectionSteps.innerHTML = '';

    const subSteps = new Set(['GroundingDINO检测', '传统CV检测', 'NMS融合去重']);

    DETECTION_STEPS.forEach((step, idx) => {
        const item = document.createElement('div');
        item.className = 'step-item';
        if (subSteps.has(step.key)) {
            item.dataset.substep = 'true';
        }

        const isCompleted = state.completedSteps.has(step.key);
        const isCurrent = state.currentStep === step.key;

        if (isCompleted) {
            item.classList.add('completed');
        } else if (isCurrent) {
            item.classList.add('active');
        }

        const icon = document.createElement('div');
        icon.className = 'step-icon';
        if (isCompleted) {
            icon.innerHTML = '&#10003;';
        } else if (isCurrent) {
            icon.innerHTML = '&#9679;';
        } else {
            icon.textContent = idx + 1;
        }

        const text = document.createElement('div');
        text.className = 'step-text';
        text.textContent = step.label;

        item.appendChild(icon);
        item.appendChild(text);
        els.detectionSteps.appendChild(item);
    });
}

function showLoading(text, isError = false) {
    els.loadingOverlay.style.display = 'flex';
    els.loadingText.textContent = text;
    els.loadingText.style.color = isError ? '#f87171' : '#94a3b8';
}

function hideLoading() {
    els.loadingOverlay.style.display = 'none';
}

function startModelStatusPolling() {
    setInterval(async () => {
        if (!state.isDetecting) {
            await loadModelStatus();
        }
    }, 2000);
}

els.btnSelect.addEventListener('click', () => {
    els.fileInput.click();
});

els.fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    try {
        els.btnSelect.disabled = true;
        els.btnStart.disabled = true;

        const result = await Api.uploadVideo(file);
        state.videoPath = result.path;
        state.videoFilename = result.filename;

        els.videoPlaceholder.style.display = 'none';
        els.video.src = await Api.getVideoStream(result.path);
        els.video.style.display = 'block';
        els.video.play().catch(() => {});

        els.btnStart.disabled = false;
        els.btnStart.textContent = '▶ 开始运行';
    } catch (e) {
        alert('上传失败: ' + e.message);
    } finally {
        els.btnSelect.disabled = false;
    }
});

els.btnStart.addEventListener('click', async () => {
    if (!state.videoPath) {
        alert('请先选择视频');
        return;
    }

    try {
        els.btnStart.disabled = true;
        els.btnStart.textContent = '启动中...';

        const result = await Api.startDetection(state.videoPath, 'cuda');
        state.pipelineId = result.pipeline_id;
        state.isDetecting = true;
        state.completedSteps.clear();
        state.currentStep = null;
        renderPipelineSteps();

        els.btnStart.style.display = 'none';
        els.btnStop.style.display = 'inline-flex';

        startStatusPolling();
        startResultsPolling();
    } catch (e) {
        alert('启动失败: ' + e.message);
        els.btnStart.disabled = false;
        els.btnStart.textContent = '▶ 开始运行';
    }
});

els.btnStop.addEventListener('click', async () => {
    if (!state.pipelineId) return;

    try {
        await Api.stopDetection(state.pipelineId);
        state.isDetecting = false;
        els.btnStart.style.display = 'inline-flex';
        els.btnStart.disabled = false;
        els.btnStart.textContent = '▶ 开始运行';
        els.btnStop.style.display = 'none';
    } catch (e) {
        alert('停止失败: ' + e.message);
    }
});

function startStatusPolling() {
    state.statusPoller = setInterval(async () => {
        if (!state.pipelineId || !state.isDetecting) {
            clearInterval(state.statusPoller);
            return;
        }

        try {
            const status = await Api.getStatus(state.pipelineId);
            updateProgress(status);

            if (status.status === 'completed' || status.status === 'stopped') {
                state.isDetecting = false;
                els.btnStart.style.display = 'inline-flex';
                els.btnStart.disabled = false;
                els.btnStart.textContent = '▶ 重新运行';
                els.btnStop.style.display = 'none';
                clearInterval(state.statusPoller);
            }
        } catch (e) {
            console.error('获取状态失败:', e);
        }
    }, 500);
}

function startResultsPolling() {
    state.resultsPoller = setInterval(async () => {
        if (!state.pipelineId || !state.isDetecting) {
            clearInterval(state.resultsPoller);
            return;
        }

        try {
            const results = await Api.getFrameResults(state.pipelineId);
            renderDetections(results.detections || []);
        } catch (e) {
            console.error('获取结果失败:', e);
        }
    }, 500);
}

function updateProgress(status) {
    els.progressBar.style.width = `${status.progress || 0}%`;
    els.progressText.textContent = `${status.progress || 0}%`;
    els.progressFrame.textContent = `帧: ${status.current_frame}/${status.total_frames}`;

    els.currentFps.textContent = status.fps ? status.fps.toFixed(1) : '-';
    els.uniqueCount.textContent = status.detections_count || 0;
    els.elapsedTime.textContent = status.elapsed_time || '-';

    if (status.last_step) {
        const matched = DETECTION_STEPS.find(s => s.key === status.last_step);
        if (matched) {
            if (state.currentStep && state.currentStep !== matched.key) {
                state.completedSteps.add(state.currentStep);
            }
            state.currentStep = matched.key;
            renderPipelineSteps();
        }
    }

    if (status.status === 'completed') {
        if (state.currentStep) {
            state.completedSteps.add(state.currentStep);
            state.currentStep = null;
            renderPipelineSteps();
        }
    }
}

function renderDetections(detections) {
    if (detections.length === 0) {
        els.detectionsTable.innerHTML = `
            <tr>
                <th>ID</th>
                <th>构件名称</th>
                <th>置信度</th>
                <th>检测器</th>
            </tr>
            <tr>
                <td colspan="4" style="text-align:center;color:#64748b;padding:20px;">暂无检测结果</td>
            </tr>
        `;
        return;
    }

    const header = `
        <tr>
            <th>ID</th>
            <th>构件名称</th>
            <th>置信度</th>
            <th>检测器</th>
        </tr>
    `;

    const rows = detections.map(d => {
        const confidenceClass = d.confidence >= 0.7 ? 'badge-green' : (d.confidence >= 0.5 ? 'badge-yellow' : 'badge-red');
        const detectorClass = d.detector === 'grounding_dino' ? 'badge-blue' : 'badge-yellow';
        const detectorName = d.detector === 'grounding_dino' ? 'GroundingDINO' : (d.detector === 'traditional_cv' ? '传统CV' : d.detector);
        return `
            <tr>
                <td>${d.track_id || '-'}</td>
                <td>${d.class_name || d.label || '-'}</td>
                <td><span class="badge ${confidenceClass}">${(d.confidence * 100).toFixed(1)}%</span></td>
                <td><span class="badge ${detectorClass}">${detectorName}</span></td>
            </tr>
        `;
    }).join('');

    els.detectionsTable.innerHTML = header + rows;
}

init();