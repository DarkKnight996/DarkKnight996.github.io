import React from 'react';

type SectionStatCardProps = {
  text: string;
};

export default function SectionStatCard({text}: SectionStatCardProps) {
  return (
    <div className="section-stat-card" role="note" aria-label="Section summary">
      <span className="section-stat-card__label">{text}</span>
    </div>
  );
}
