function result = ecg_preprocess(ecg, fs)
%ECG_PREPROCESS Offline ECG preprocessing with a 50 Hz notch and 0.5-40 Hz bandpass.
%   RESULT = ECG_PREPROCESS(ECG, FS) takes a single-channel ECG vector and
%   its sampling rate in Hz. RESULT.raw and RESULT.filtered preserve the
%   input shape and physical units. RESULT.fs records the sampling rate.
%   The raw signal is preserved; 1.65 V is subtracted before filtering.
%
%   The 50 Hz notch has Q = 30. The bandpass uses a fourth-order
%   Butterworth prototype. Both filters are applied forward and backward,
%   so this function is intended for complete recordings, not streaming.
%   Signal Processing Toolbox is required. Filtering may alter ECG
%   morphology, particularly near the beginning and end of a recording.

    raw = ecg;
    x = double(ecg(:)) - 1.65;
    fs = double(fs);

    % Equivalent second-order notch design with bandwidth 50/Q Hz.
    notchHz = 50;
    notchQ = 30;
    omega = 2 * pi * notchHz / fs;
    beta = tan(omega / (2 * notchQ));
    gain = 1 / (1 + beta);
    bNotch = gain * [1, -2 * cos(omega), 1];
    aNotch = [1, -2 * gain * cos(omega), 2 * gain - 1];

    % Second-order sections improve numerical stability for the bandpass.
    [z, p, k] = butter(4, [0.5, 40] / (fs / 2), 'bandpass');
    [sos, g] = zp2sos(z, p, k);

    x = filtfilt(bNotch, aNotch, x);
    x = filtfilt(sos, g, x);

    result.raw = raw;
    result.filtered = reshape(x, size(raw));
    result.fs = fs;
end
