import React from 'react';

type PaperCardProps = {
  title: string;
  link?: string;
  authors?: string;
  institution?: string;
  summary?: string;
  tags?: string;
};

function parseTags(tags: string | undefined): string[] {
  if (!tags) {
    return [];
  }

  try {
    const parsed = JSON.parse(tags);
    return Array.isArray(parsed)
      ? parsed.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
      : [];
  } catch {
    return [];
  }
}

export default function PaperCard({
  title,
  link,
  authors,
  institution,
  summary,
  tags,
}: PaperCardProps) {
  const parsedTags = parseTags(tags);

  return (
    <article className="paper-card">
      <header className="paper-card__header">
        <div className="paper-card__title-block">
          <h3 className="paper-card__title">{title}</h3>
        </div>
        {link ? (
          <a
            className="paper-card__arxiv-link"
            href={link}
            target="_blank"
            rel="noreferrer noopener">
            arXiv
          </a>
        ) : null}
        {parsedTags.length > 0 ? (
          <div className="paper-card__tags" aria-label="Paper tags">
            {parsedTags.map((tag) => (
              <span key={tag} className="paper-card__tag">
                {tag}
              </span>
            ))}
          </div>
        ) : null}
      </header>

      <dl className="paper-card__meta">
        {authors ? (
          <>
            <dt>Authors</dt>
            <dd>{authors}</dd>
          </>
        ) : null}
        {institution ? (
          <>
            <dt>Institution</dt>
            <dd>{institution}</dd>
          </>
        ) : null}
      </dl>

      {summary ? <p className="paper-card__summary">{summary}</p> : null}
    </article>
  );
}
