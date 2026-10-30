function locs = ecg_pan_tompkins(ecg, fs)
%ECG_PAN_TOMPKINS Detect ECG peaks with an offline Pan-Tompkins-style method.
%   LOCS = ECG_PAN_TOMPKINS(ECG, FS) returns 1-based sample indices in the
%   input signal. ECG must be a finite single-channel filtered ECG vector.
%   The implementation uses a QRS bandpass, derivative, squaring, moving
%   integration, adaptive thresholds, a refractory period and search-back.
%   Candidate positions are refined on ECG to account for either polarity.
%   Signal Processing Toolbox is required. This offline implementation is
%   not a real-time or clinically validated detector.

    if ~isnumeric(ecg) || ~isreal(ecg) || ~isvector(ecg) || ...
            isempty(ecg) || any(~isfinite(ecg(:)))
        error('ecg_pan_tompkins:InvalidSignal', ...
            'ECG must be a nonempty, real, finite numeric vector.');
    end
    if ~isnumeric(fs) || ~isscalar(fs) || ~isreal(fs) || ...
            ~isfinite(fs) || fs <= 40
        error('ecg_pan_tompkins:InvalidSampleRate', ...
            'Sampling rate must be a finite scalar greater than 40 Hz.');
    end
    x = double(ecg(:));
    fs = double(fs);
    locs = zeros(0, 1);
    if numel(x) < max(20, round(0.5 * fs)) || max(x) == min(x)
        return;
    end

    % Offline QRS enhancement, followed by the standard energy envelope.
    [b, a] = butter(2, [5, 15] / (fs / 2), 'bandpass');
    qrs = filtfilt(b, a, x);
    derivative = conv(qrs, [1 2 0 -2 -1] * (fs / 8), 'same');
    width = max(1, round(0.15 * fs));
    envelope = conv(derivative .^ 2, ones(width, 1) / width, 'same');
    if max(envelope) <= 0
        return;
    end

    refractory = max(1, round(0.20 * fs));
    [values, positions] = findpeaks(envelope, 'MinPeakDistance', refractory);
    if isempty(positions)
        return;
    end
    initial = envelope(1:min(numel(x), max(1, round(2 * fs))));
    signalLevel = max(initial);
    noiseLevel = min(mean(initial), signalLevel);
    accepted = zeros(0, 1);
    pendingPos = zeros(0, 1);
    pendingVal = zeros(0, 1);
    lastSlope = 0;
    slopeRadius = max(1, round(0.075 * fs));

    for i = 1:numel(positions)
        pos = positions(i);
        value = values(i);
        threshold = noiseLevel + 0.25 * (signalLevel - noiseLevel);

        % Recover a missed beat when the interval exceeds recent RR history.
        if numel(accepted) >= 3
            intervals = diff(accepted(max(1, end-8):end));
            if pos - accepted(end) > 1.66 * mean(intervals)
                eligible = pendingPos > accepted(end) + refractory & ...
                    pendingPos < pos - refractory & pendingVal >= 0.5 * threshold;
                if any(eligible)
                    eligibleIndices = find(eligible);
                    [~, best] = max(pendingVal(eligible));
                    recovered = eligibleIndices(best);
                    accepted(end+1, 1) = pendingPos(recovered); %#ok<AGROW>
                    signalLevel = 0.25 * pendingVal(recovered) + 0.75 * signalLevel;
                    left = max(1, accepted(end) - slopeRadius);
                    right = min(numel(x), accepted(end) + slopeRadius);
                    lastSlope = max(abs(derivative(left:right)));
                    threshold = noiseLevel + 0.25 * (signalLevel - noiseLevel);
                end
            end
        end

        left = max(1, pos - slopeRadius);
        right = min(numel(x), pos + slopeRadius);
        slope = max(abs(derivative(left:right)));
        isTWave = ~isempty(accepted) && pos - accepted(end) < round(0.36 * fs) ...
            && slope < 0.5 * lastSlope;
        if value >= threshold && ~isTWave && ...
                (isempty(accepted) || pos - accepted(end) >= refractory)
            accepted(end+1, 1) = pos; %#ok<AGROW>
            signalLevel = 0.125 * value + 0.875 * signalLevel;
            lastSlope = slope;
        else
            noiseLevel = 0.125 * value + 0.875 * noiseLevel;
            pendingPos(end+1, 1) = pos; %#ok<AGROW>
            pendingVal(end+1, 1) = value; %#ok<AGROW>
        end
    end

    % Find the strongest deflection in the original filtered ECG near each
    % envelope peak. A local median permits positive or negative QRS peaks.
    radius = max(1, round(0.15 * fs));
    for i = 1:numel(accepted)
        left = max(1, accepted(i) - radius);
        right = min(numel(x), accepted(i) + radius);
        segment = x(left:right);
        [~, offset] = max(abs(segment - median(segment)));
        peak = left + offset - 1;
        if isempty(locs) || peak - locs(end) >= refractory
            locs(end+1, 1) = peak; %#ok<AGROW>
        end
    end
end
