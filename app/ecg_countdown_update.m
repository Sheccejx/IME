function ecg_countdown_update(label, session, event)
%ECG_COUNTDOWN_UPDATE Refresh remaining acquisition time from DAQ timestamps.
%   The last timestamp corresponds to the start of its sample interval.
    if ~isvalid(label) || isempty(event.TimeStamps)
        return;
    end
    acquiredTime = event.TimeStamps(end) + 1 / session.Rate;
    remaining = max(0, ceil(session.DurationInSeconds - acquiredTime));
    label.Text = sprintf('Acquiring: %d s remaining', remaining);
    drawnow limitrate;
end
