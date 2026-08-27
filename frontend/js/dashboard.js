let state = {
    modelStatus: null,
    pipelineId: null,
    videoPath: null,
    videoFilename: null,
    isDetecting: false,
    statusPoller: null,
    resultsPoller: null,
    displayedComponentIds: new Set(),
    detectionProgress: 0,
    currentFrameImage: null,
    currentFrameNum: 0,
    totalFrames: 0,
    videoFps: 30,
};

const els = {
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),
    videoContainer: document.getElementById('videoContainer'),
    video: document.getElementById('video'),
    videoPlaceholder: document.getElementById('videoPlaceholder'),
    frameOverlay: document.getElementById('frameOverlay'),
    frameOverlayImg: document.getElementById('frameOverlayImg'),
    frameOverlayLabel: document.getElementById('frameOverlayLabel'),
    fileInput: document.getElementById('fileInput'),
    btnSelect: document.getElementById('btnSelect'),
    btnStart: document.getElementById('btnStart'),
    btnStop: document.getElementById('btnStop'),
    progressBar: document.getElementById('progressBar'),
    progressFill: document.getElementById('progressFill'),
    progressFrameMarker: document.getElementById('progressFrameMarker'),
    progressHover: document.getElementById('progressHover'),
    progressTooltip: document.getElementById('progressTooltip'),
    progressText: document.getElementById('progressText'),
    progressFrame: document.getElementById('progressFrame'),
    progressTime: document.getElementById('progressTime'),
    progressPercent: document.getElementById('progressPercent'),
    pipelineResults: document.getElementById('pipelineResults'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingText: document.getElementById('loadingText'),
    modelInitialized: document.getElementById('modelInitialized'),
    modelGroundingDino: document.getElementById('modelGroundingDino'),
    modelClip: document.getElementById('modelClip'),
    modelLoadProgress: document.getElementById('modelLoadProgress'),
    uniqueCount: document.getElementById('uniqueCount'),
    currentFps: document.getElementById('currentFps'),
    elapsedTime: document.getElementById('elapsedTime'),
    lightbox: document.getElementById('lightbox'),
    lightboxImg: document.getElementById('lightboxImg'),
    lightboxClose: document.getElementById('lightboxClose'),
};

async function init() {
    try {
        await Api.healthCheck();
        updateStatus('loading', '加载中...');
        await loadModelStatus();
        startModelStatusPolling();
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
        updateStartButtonState();

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

function updateStartButtonState() {
    const modelReady = state.modelStatus && state.modelStatus.initialized;
    const hasVideo = !!state.videoPath;
    if (!state.isDetecting && hasVideo) {
        els.btnStart.disabled = !modelReady;
        els.btnStart.title = modelReady ? '开始运行检测' : '模型加载中，请稍候...';
    } else if (!hasVideo) {
        els.btnStart.disabled = true;
        els.btnStart.title = '请先选择视频';
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

        updateStartButtonState();
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

    if (!state.modelStatus || !state.modelStatus.initialized) {
        alert('模型尚未加载完成，请稍候');
        return;
    }

    try {
        els.btnStart.disabled = true;
        els.btnStart.textContent = '启动中...';

        state.detectionProgress = 0;
        state.currentFrameNum = 0;
        state.currentFrameImage = null;
        state.totalFrames = 0;
        els.progressFill.style.width = '0%';
        els.progressFrameMarker.style.left = '0%';
        els.progressText.textContent = '0%';
        els.progressPercent.textContent = '0%';

        els.video.pause();
        els.video.style.display = 'none';
        els.frameOverlayImg.src = '';
        els.frameOverlayLabel.textContent = '初始化中...';
        els.frameOverlay.style.display = 'flex';

        const result = await Api.startDetection(state.videoPath, 'cuda');
        state.pipelineId = result.pipeline_id;
        state.isDetecting = true;
        state.displayedComponentIds.clear();
        els.pipelineResults.innerHTML = '<div class="pipeline-empty">检测中，正在识别构件...</div>';

        els.btnStart.style.display = 'none';
        els.btnStop.style.display = 'inline-flex';

        startStatusPolling();
        startResultsPolling();
    } catch (e) {
        els.frameOverlay.style.display = 'none';
        els.video.style.display = 'block';
        alert('启动失败: ' + e.message);
        els.btnStart.disabled = false;
        els.btnStart.textContent = '▶ 重新运行';
    }
});

els.btnStop.addEventListener('click', async () => {
    if (!state.pipelineId) return;

    try {
        await Api.stopDetection(state.pipelineId);
        state.isDetecting = false;
        els.btnStart.style.display = 'inline-flex';
        els.btnStart.textContent = '▶ 重新运行';
        els.btnStop.style.display = 'none';
        els.frameOverlay.style.display = 'none';
        els.video.style.display = 'block';
        updateStartButtonState();
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
                els.btnStart.textContent = '▶ 重新运行';
                els.btnStop.style.display = 'none';
                els.frameOverlay.style.display = 'none';
                els.video.style.display = 'block';
                updateStartButtonState();
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

            if (results && results.image_filename) {
                const url = `/api/pipeline/annotated_image?filename=${encodeURIComponent(results.image_filename)}&t=${Date.now()}`;
                state.currentFrameImage = url;
                if (state.isDetecting) {
                    els.frameOverlayImg.src = url;
                }
            }
        } catch (e) {
            console.error('获取结果失败:', e);
        }
    }, 500);
}

function formatTime(seconds) {
    if (!isFinite(seconds) || seconds < 0) return '00:00';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function updateProgress(status) {
    const progress = status.progress || 0;
    state.detectionProgress = progress;
    state.currentFrameNum = status.current_frame || 0;
    state.totalFrames = status.total_frames || 0;
    state.videoFps = status.fps || 30;

    els.progressFill.style.width = `${progress}%`;
    els.progressFrameMarker.style.left = `${progress}%`;
    els.progressText.textContent = `${progress}%`;
    els.progressPercent.textContent = `${progress}%`;

    const frameText = state.totalFrames > 0
        ? `帧: ${state.currentFrameNum}/${state.totalFrames}`
        : `帧: ${state.currentFrameNum}`;
    els.progressFrame.textContent = frameText;

    const currentTime = state.currentFrameNum / state.videoFps;
    const totalTime = state.totalFrames > 0 ? state.totalFrames / state.videoFps : (els.video.duration || 0);
    els.progressTime.textContent = `${formatTime(currentTime)} / ${formatTime(totalTime)}`;

    els.currentFps.textContent = status.fps ? status.fps.toFixed(1) : '-';
    els.uniqueCount.textContent = status.detections_count || 0;
    els.elapsedTime.textContent = status.elapsed_time || '-';

    if (state.isDetecting && state.currentFrameImage) {
        els.frameOverlayImg.src = state.currentFrameImage;
        els.frameOverlayLabel.textContent = `帧 ${state.currentFrameNum} / ${state.totalFrames}`;
    }
}

function renderDetections(detections) {
    if (detections.length === 0) {
        if (state.displayedComponentIds.size === 0) {
            els.pipelineResults.innerHTML = '<div class="pipeline-empty">暂无检测结果</div>';
        }
        return;
    }

    const newComponents = detections.filter(d => !state.displayedComponentIds.has(d.component_id));
    if (newComponents.length === 0) {
        return;
    }

    newComponents.forEach(d => state.displayedComponentIds.add(d.component_id));

    if (state.displayedComponentIds.size === detections.length) {
        els.pipelineResults.innerHTML = '';
    }

    const cardsHtml = newComponents.map(d => {
        const confidenceClass = d.confidence >= 0.7 ? 'badge-green' : (d.confidence >= 0.5 ? 'badge-yellow' : 'badge-red');
        const detectorClass = d.detector === 'grounding_dino' ? 'badge-blue' : 'badge-yellow';
        const detectorName = d.detector === 'grounding_dino' ? 'GroundingDINO' : (d.detector === 'traditional_cv' ? '传统CV' : (d.detector || '-'));

        const baseUrl = '/api/pipeline/annotated_image?filename=';

        const stepCrane = d.crane_image_filename
            ? `<div class="pipeline-step-image" data-src="${baseUrl}${encodeURIComponent(d.crane_image_filename)}">
                 <span class="pipeline-frame-badge">帧 ${d.frame}</span>
                 <img src="${baseUrl}${encodeURIComponent(d.crane_image_filename)}" alt="吊机检测" loading="lazy">
               </div>`
            : `<div class="pipeline-step-placeholder">吊机检测<br>无数据</div>`;

        const stepROI = d.roi_image_filename
            ? `<div class="pipeline-step-image" data-src="${baseUrl}${encodeURIComponent(d.roi_image_filename)}">
                 <span class="pipeline-frame-badge">帧 ${d.frame}</span>
                 <img src="${baseUrl}${encodeURIComponent(d.roi_image_filename)}" alt="检测范围" loading="lazy">
               </div>`
            : `<div class="pipeline-step-placeholder">检测范围<br>无数据</div>`;

        const stepDetect = d.image_filename
            ? `<div class="pipeline-step-image" data-src="${baseUrl}${encodeURIComponent(d.image_filename)}">
                 <span class="pipeline-frame-badge">帧 ${d.frame}</span>
                 <img src="${baseUrl}${encodeURIComponent(d.image_filename)}" alt="构件检测" loading="lazy">
               </div>`
            : `<div class="pipeline-step-placeholder">构件检测<br>无数据</div>`;

        const stepCompROI = d.component_roi_filename
            ? `<div class="pipeline-step-image" data-src="${baseUrl}${encodeURIComponent(d.component_roi_filename)}">
                 <span class="pipeline-frame-badge">#${d.component_id ? d.component_id.split('_').pop() : ''}</span>
                 <img src="${baseUrl}${encodeURIComponent(d.component_roi_filename)}" alt="构件ROI" loading="lazy">
               </div>`
            : `<div class="pipeline-step-placeholder">构件ROI<br>无数据</div>`;

        return `
            <div class="pipeline-card">
                <div class="pipeline-header">
                    <div class="pipeline-title">
                        <span>${d.class_name || d.label || '未知构件'}</span>
                        <span class="pipeline-id-tag">${d.component_id || d.track_id || ''}</span>
                    </div>
                    <div style="display:flex;gap:8px;align-items:center;">
                        <span class="badge ${detectorClass}">${detectorName}</span>
                        <span class="badge ${confidenceClass}">${(d.confidence * 100).toFixed(1)}%</span>
                    </div>
                </div>
                <div class="pipeline-steps">
                    <div class="pipeline-step">
                        <div class="pipeline-step-label"><span class="step-icon-inline">1</span>吊机检测</div>
                        ${stepCrane}
                    </div>
                    <div class="pipeline-step">
                        <div class="pipeline-step-label"><span class="step-icon-inline">2</span>检测范围</div>
                        ${stepROI}
                    </div>
                    <div class="pipeline-step">
                        <div class="pipeline-step-label"><span class="step-icon-inline">3</span>构件检测</div>
                        ${stepDetect}
                    </div>
                    <div class="pipeline-step">
                        <div class="pipeline-step-label"><span class="step-icon-inline">4</span>构件ROI</div>
                        ${stepCompROI}
                    </div>
                </div>
                <div class="pipeline-meta">
                    <div class="pipeline-meta-item">帧号: ${d.frame}</div>
                    <div class="pipeline-meta-item">ID: ${d.track_id || '-'}</div>
                </div>
            </div>
        `;
    }).join('');

    els.pipelineResults.insertAdjacentHTML('beforeend', cardsHtml);

    els.pipelineResults.querySelectorAll('.pipeline-step-image').forEach(el => {
        el.addEventListener('click', () => {
            const src = el.dataset.src;
            if (src) {
                els.lightboxImg.src = src;
                els.lightbox.style.display = 'flex';
            }
        });
    });
}

els.video.addEventListener('loadedmetadata', () => {
    if (els.video.duration && isFinite(els.video.duration)) {
        if (!state.totalFrames || state.totalFrames === 0) {
            state.totalFrames = Math.round(els.video.duration * (state.videoFps || 30));
        }
    }
});

els.progressBar.addEventListener('mousemove', (e) => {
    const rect = els.progressBar.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));

    els.progressHover.style.display = 'block';
    els.progressHover.style.left = `${pct}%`;

    let tooltipText;
    if (state.totalFrames > 0) {
        const frame = Math.round((pct / 100) * state.totalFrames);
        const time = frame / state.videoFps;
        tooltipText = `帧 ${frame}/${state.totalFrames} | ${formatTime(time)}`;
    } else {
        const time = (pct / 100) * (els.video.duration || 0);
        tooltipText = `${formatTime(time)}`;
    }
    els.progressTooltip.textContent = tooltipText;
    els.progressTooltip.style.display = 'block';
    els.progressTooltip.style.left = `${pct}%`;
});

els.progressBar.addEventListener('mouseleave', () => {
    els.progressHover.style.display = 'none';
    els.progressTooltip.style.display = 'none';
});

els.progressBar.addEventListener('click', (e) => {
    if (state.isDetecting) return;

    const rect = els.progressBar.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));

    if (state.totalFrames > 0) {
        const frame = Math.round((pct / 100) * state.totalFrames);
        const time = frame / state.videoFps;
        if (els.video.duration) {
            els.video.currentTime = Math.min(time, els.video.duration - 0.1);
        }
    }
});

els.lightboxClose.addEventListener('click', () => {
    els.lightbox.style.display = 'none';
});

els.lightbox.addEventListener('click', (e) => {
    if (e.target === els.lightbox) {
        els.lightbox.style.display = 'none';
    }
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && els.lightbox.style.display === 'flex') {
        els.lightbox.style.display = 'none';
    }
});

init();