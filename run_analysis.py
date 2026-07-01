import subprocess, sys

# Install missing dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install", "textblob", "nltk", "-q"])

import pandas as pd
import string
import nltk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# ── Helpers ──────────────────────────────────────────────────────────────────
def clean_text(text):
    if pd.isnull(text):
        return ""
    tokens = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    return ' '.join([w for w in tokens if w not in stop_words])

def label_sentiment(score):
    if score > 0.1:
        return 'positive'
    elif score < -0.1:
        return 'negative'
    else:
        return 'neutral'

# ── Step 1: Build raw datasets ────────────────────────────────────────────────
glassdoor_raw = [
    {"summary": "Pay fair but warehouse labor is exhausting"},
    {"summary": "Managers micromanage all daily tasks"},
    {"summary": "Great holiday benefits and flexible leave"},
]
youtube_raw = [
    {"transcript": "12-hour shifts drain my physical energy every day"},
    {"transcript": "No clear permanent full-time hiring paths"},
    {"transcript": "Overtime wage is the only advantage of this job"},
]

def build_df(records, text_col, output_col):
    df = pd.DataFrame(records)
    df[output_col] = df[text_col].apply(clean_text)
    df['sentiment_polarity'] = df[output_col].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentiment_subjectivity'] = df[output_col].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    df['sentiment_label'] = df['sentiment_polarity'].apply(label_sentiment)
    return df

glassdoor_df = build_df(glassdoor_raw, 'summary', 'cleaned_text')
glassdoor_df['platform'] = 'Glassdoor'

youtube_df = build_df(youtube_raw, 'transcript', 'cleaned_text')
youtube_df['platform'] = 'YouTube'

# ── Step 2: Combine & add themes ──────────────────────────────────────────────
combined_df = pd.concat([glassdoor_df, youtube_df], ignore_index=True)

theme_keywords = {
    "Shift & Burnout Issues": ["tired", "exhausted", "burnout", "overworked", "no break",
                               "long shift", "night shift", "schedule", "shift", "drain", "energy"],
    "Onboarding & Training Gaps": ["training", "onboarding", "new hire", "no support",
                                    "confused", "untrained", "learn", "hiring", "paths"],
    "Micromanagement & Manager Conflicts": ["manager", "micromanage", "supervisor",
                                             "unfair", "leadership", "strict"],
}

def tag_theme(text):
    text_lower = str(text).lower()
    matches = [theme for theme, words in theme_keywords.items()
               if any(w in text_lower for w in words)]
    return matches[0] if matches else "Other"

combined_df['Theme'] = combined_df['cleaned_text'].apply(tag_theme)
combined_df.to_csv('C:\\Users\\DELL\\Desktop\\zsz\\combined_with_themes.csv', index=False)
print("Saved: combined_with_themes.csv")
print(combined_df[["cleaned_text", "sentiment_label", "platform", "Theme"]].to_string())

# ── Step 3: Grouped bar chart — sentiment by theme ────────────────────────────
df = combined_df.copy()

# Only rows with a real theme (exclude "Other" if desired — keep all here)
theme_sentiment = (
    df.groupby(['Theme', 'sentiment_label'])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=['positive', 'neutral', 'negative'], fill_value=0)
)

colors = {'positive': '#4CAF50', 'neutral': '#FFC107', 'negative': '#F44336'}
fig, ax = plt.subplots(figsize=(11, 6))

x = range(len(theme_sentiment))
width = 0.25
labels = ['Positive', 'Neutral', 'Negative']
keys = ['positive', 'neutral', 'negative']

for i, (key, label) in enumerate(zip(keys, labels)):
    bars = ax.bar(
        [xi + i * width for xi in x],
        theme_sentiment[key],
        width=width,
        label=label,
        color=colors[key],
        edgecolor='white',
    )
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, str(int(h)),
                    ha='center', va='bottom', fontsize=9)

ax.set_xticks([xi + width for xi in x])
ax.set_xticklabels(theme_sentiment.index, fontsize=9, wrap=True)
ax.set_xlabel('Theme', fontsize=11)
ax.set_ylabel('Number of Reviews', fontsize=11)
ax.set_title('Sentiment Distribution by Theme\n(Amazon Warehouse Employee Feedback)', fontsize=13, fontweight='bold')
ax.legend(title='Sentiment')
ax.set_ylim(0, max(theme_sentiment.values.max() + 1, 3))
plt.tight_layout()
out1 = 'C:\\Users\\DELL\\Desktop\\zsz\\sentiment_by_theme.png'
plt.savefig(out1, dpi=150)
print(f"Chart saved: {out1}")
plt.close()

# ── Step 4: Cross-platform comparison chart ───────────────────────────────────
platform_theme = (
    df.groupby(['Theme', 'platform', 'sentiment_label'])
    .size()
    .reset_index(name='count')
)

themes = df['Theme'].unique()
platforms = ['Glassdoor', 'YouTube']
sentiments = ['positive', 'neutral', 'negative']
sent_colors = {'positive': '#4CAF50', 'neutral': '#FFC107', 'negative': '#F44336'}

fig2, axes = plt.subplots(1, len(themes), figsize=(6 * len(themes), 6), sharey=True)
if len(themes) == 1:
    axes = [axes]

for ax2, theme in zip(axes, themes):
    sub = platform_theme[platform_theme['Theme'] == theme]
    pivot = sub.pivot_table(index='platform', columns='sentiment_label', values='count', fill_value=0)
    pivot = pivot.reindex(columns=['positive', 'neutral', 'negative'], fill_value=0)
    pivot = pivot.reindex(platforms, fill_value=0)

    x2 = range(len(pivot))
    for i, sent in enumerate(sentiments):
        if sent in pivot.columns:
            bars2 = ax2.bar(
                [xi + i * 0.25 for xi in x2],
                pivot[sent],
                width=0.25,
                label=sent.capitalize(),
                color=sent_colors[sent],
                edgecolor='white',
            )
            for bar in bars2:
                h = bar.get_height()
                if h > 0:
                    ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.02, str(int(h)),
                             ha='center', va='bottom', fontsize=8)

    ax2.set_xticks([xi + 0.25 for xi in x2])
    ax2.set_xticklabels(pivot.index, fontsize=10)
    ax2.set_title(theme, fontsize=9, fontweight='bold', wrap=True)
    ax2.set_xlabel('Platform')

axes[0].set_ylabel('Number of Reviews', fontsize=11)
handles, lbls = axes[0].get_legend_handles_labels()
fig2.legend(handles, lbls, title='Sentiment', loc='upper right')
fig2.suptitle('Cross-Platform Sentiment Comparison by Theme', fontsize=13, fontweight='bold')
plt.tight_layout()
out2 = 'C:\\Users\\DELL\\Desktop\\zsz\\sentiment_by_theme_platform.png'
plt.savefig(out2, dpi=150)
print(f"Chart saved: {out2}")
plt.close()

print("\nAll done.")
