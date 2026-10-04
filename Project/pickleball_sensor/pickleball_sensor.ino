/*
  Pickleball Piezo Event Capture - Restored Simple Monitor
  ========================================================

  Hardware (freeze this while testing):
    Piezo signal -> 100k series resistor -> A0
    A0 -> 1M bias resistor -> VREF (~2.5V)
    Silicon clamp diodes remain installed
    A1 -> VREF

  Why this trigger:
    Your measured idle vibration is ~0-2 ADC peak-to-peak.
    Ball impacts on the desk were typically much larger, with the weakest
    controlled 1M trial reaching about 4 ADC peak-to-peak.

    Therefore this detector looks at the RANGE of the last 32 samples:
        max(recent) - min(recent)

    Slow baseline drift does not matter because it is not measuring
    absolute distance from a stale baseline.

  Serial protocol:
    STATUS,READY,sensor,vref
    LIVE,time_us,sensor,vref,recent_pp
    EVENT,event_id,event_baseline,sample_period_us,raw0,...,raw255
*/

const int SENSOR_PIN = A0;
const int VREF_PIN   = A1;

const unsigned long SERIAL_BAUD = 1000000;

// Trigger from short-window peak-to-peak movement.
// Idle was ~0-2 ADC p-p; use 4 as the measured starting point.
const int TRIGGER_PP_THRESHOLD = 4;

// Last ~32 ADC samples used to detect vibration.
// At ~9 kHz this is around 3-4 ms.
const int TRIGGER_WINDOW = 32;

// Full event waveform.
const int NUM_EVENT_SAMPLES = 256;

// The circular pre-trigger buffer.
// The newest pre-trigger sample is the sample that completed the trigger.
const int PRE_TRIGGER_SAMPLES = 64;

// Avoid one ringing impact generating several events.
const unsigned long EVENT_COOLDOWN_MS = 100;

// Diagnostic output only.
const unsigned long LIVE_INTERVAL_US = 10000; // 100 Hz


// ============================================================
// BUFFERS
// ============================================================

int triggerHistory[TRIGGER_WINDOW];
int triggerWriteIndex = 0;
int triggerCount = 0;

int preRaw[PRE_TRIGGER_SAMPLES];
int preWriteIndex = 0;
int preCount = 0;

int eventRaw[NUM_EVENT_SAMPLES];


// ============================================================
// STATE
// ============================================================

unsigned long eventId = 0;
unsigned long lastEventMs = 0;
unsigned long lastLiveUs = 0;


// ============================================================
// BASIC HELPERS
// ============================================================

int readVrefDiagnostic()
{
    // Mega ADC is multiplexed. Settle after changing channel.
    analogRead(VREF_PIN);
    int vref = analogRead(VREF_PIN);

    // Switch back to A0 and discard one conversion.
    analogRead(SENSOR_PIN);

    return vref;
}


int averageSensor(int count)
{
    long sum = 0;

    analogRead(SENSOR_PIN);

    for (int i = 0; i < count; ++i)
        sum += analogRead(SENSOR_PIN);

    return (count > 0) ? (int)(sum / count) : 0;
}


// ============================================================
// TRIGGER WINDOW
// ============================================================

void clearTriggerHistory(int value)
{
    triggerWriteIndex = 0;
    triggerCount = TRIGGER_WINDOW;

    for (int i = 0; i < TRIGGER_WINDOW; ++i)
        triggerHistory[i] = value;
}


void pushTriggerHistory(int raw)
{
    triggerHistory[triggerWriteIndex] = raw;

    triggerWriteIndex++;
    if (triggerWriteIndex >= TRIGGER_WINDOW)
        triggerWriteIndex = 0;

    if (triggerCount < TRIGGER_WINDOW)
        triggerCount++;
}


int recentPeakToPeak()
{
    if (triggerCount <= 1)
        return 0;

    int minimum = 1023;
    int maximum = 0;

    for (int i = 0; i < triggerCount; ++i)
    {
        int value = triggerHistory[i];

        if (value < minimum)
            minimum = value;

        if (value > maximum)
            maximum = value;
    }

    return maximum - minimum;
}


// ============================================================
// PRE-TRIGGER BUFFER
// ============================================================

void clearPreTrigger(int value)
{
    preWriteIndex = 0;
    preCount = PRE_TRIGGER_SAMPLES;

    for (int i = 0; i < PRE_TRIGGER_SAMPLES; ++i)
        preRaw[i] = value;
}


void pushPreTrigger(int raw)
{
    preRaw[preWriteIndex] = raw;

    preWriteIndex++;
    if (preWriteIndex >= PRE_TRIGGER_SAMPLES)
        preWriteIndex = 0;

    if (preCount < PRE_TRIGGER_SAMPLES)
        preCount++;
}


void copyPreTriggerToEvent()
{
    int missing = PRE_TRIGGER_SAMPLES - preCount;

    int fallback = 512;

    if (preCount > 0)
    {
        int oldest =
            (preWriteIndex - preCount + PRE_TRIGGER_SAMPLES)
            % PRE_TRIGGER_SAMPLES;

        fallback = preRaw[oldest];
    }

    for (int i = 0; i < missing; ++i)
        eventRaw[i] = fallback;

    int oldest =
        (preWriteIndex - preCount + PRE_TRIGGER_SAMPLES)
        % PRE_TRIGGER_SAMPLES;

    for (int i = 0; i < preCount; ++i)
    {
        int src = (oldest + i) % PRE_TRIGGER_SAMPLES;
        eventRaw[missing + i] = preRaw[src];
    }
}


// ============================================================
// EVENT BASELINE
// ============================================================

int calculateEventBaseline()
{
    // Use the first half of the pre-trigger history.
    // This is intentionally before the trigger region so the impact
    // itself does not shift the zero level.
    const int COUNT = PRE_TRIGGER_SAMPLES / 2;

    long sum = 0;

    for (int i = 0; i < COUNT; ++i)
        sum += eventRaw[i];

    return (int)(sum / COUNT);
}


// ============================================================
// EVENT TRANSMISSION
// ============================================================

void sendEvent(int baseline, float samplePeriodUs)
{
    Serial.print("EVENT,");
    Serial.print(eventId);
    Serial.print(",");
    Serial.print(baseline);
    Serial.print(",");
    Serial.print(samplePeriodUs, 3);

    for (int i = 0; i < NUM_EVENT_SAMPLES; ++i)
    {
        Serial.print(",");
        Serial.print(eventRaw[i]);
    }

    Serial.println();
}


// ============================================================
// EVENT CAPTURE
// ============================================================

void captureEvent()
{
    eventId++;
    lastEventMs = millis();

    // The pre-buffer already contains the samples that caused the trigger.
    copyPreTriggerToEvent();

    // Capture remaining samples immediately with NO serial output.
    unsigned long captureStartUs = micros();

    for (int i = PRE_TRIGGER_SAMPLES; i < NUM_EVENT_SAMPLES; ++i)
        eventRaw[i] = analogRead(SENSOR_PIN);

    unsigned long captureEndUs = micros();

    int postSamples =
        NUM_EVENT_SAMPLES - PRE_TRIGGER_SAMPLES;

    float samplePeriodUs = 112.0f;

    if (postSamples > 1)
    {
        samplePeriodUs =
            (float)(captureEndUs - captureStartUs)
            / (float)postSamples;
    }

    int baseline = calculateEventBaseline();

    sendEvent(baseline, samplePeriodUs);

    // Reset both windows to the CURRENT resting level.
    // This prevents previous event ringing from contaminating the next trigger.
    int resting = averageSensor(64);

    clearTriggerHistory(resting);
    clearPreTrigger(resting);
}


// ============================================================
// SETUP
// ============================================================

void setup()
{
    Serial.begin(SERIAL_BAUD);
    delay(1000);

    int resting = averageSensor(200);
    int vref = readVrefDiagnostic();

    clearTriggerHistory(resting);
    clearPreTrigger(resting);

    Serial.print("STATUS,READY,");
    Serial.print(resting);
    Serial.print(",");
    Serial.println(vref);
}


// ============================================================
// MAIN LOOP
// ============================================================

void loop()
{
    unsigned long nowUs = micros();

    int raw = analogRead(SENSOR_PIN);

    // Always retain the real waveform history.
    pushPreTrigger(raw);
    pushTriggerHistory(raw);

    int recentPP = recentPeakToPeak();

    bool cooldownComplete =
        (millis() - lastEventMs) >= EVENT_COOLDOWN_MS;

    if (
        cooldownComplete
        &&
        recentPP >= TRIGGER_PP_THRESHOLD
    )
    {
        captureEvent();
        return;
    }

    // Low-rate diagnostics only. Never used for triggering.
    if ((nowUs - lastLiveUs) >= LIVE_INTERVAL_US)
    {
        lastLiveUs = nowUs;

        int vref = readVrefDiagnostic();

        Serial.print("LIVE,");
        Serial.print(nowUs);
        Serial.print(",");
        Serial.print(raw);
        Serial.print(",");
        Serial.print(vref);
        Serial.print(",");
        Serial.println(recentPP);
    }
}
