import React, {type ReactNode, useEffect, useRef, useState} from 'react';
import clsx from 'clsx';
import {ThemeClassNames} from '@docusaurus/theme-common';
import {useDoc} from '@docusaurus/plugin-content-docs/client';
import TOC from '@theme/TOC';

import styles from './styles.module.css';

export default function DocItemTOCDesktop(): ReactNode {
  const {toc, frontMatter} = useDoc();
  const [isOpen, setIsOpen] = useState(false);
  const panelRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!isOpen) {
      return undefined;
    }

    function handlePointerDown(event: MouseEvent) {
      if (!panelRef.current?.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        setIsOpen(false);
      }
    }

    document.addEventListener('mousedown', handlePointerDown);
    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('mousedown', handlePointerDown);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen]);

  function scrollToComments() {
    const commentsSection = document.getElementById('comments-section');
    commentsSection?.scrollIntoView({behavior: 'smooth', block: 'start'});
    setIsOpen(false);
  }

  return (
    <div
      ref={panelRef}
      className={clsx(styles.tocShell, isOpen && styles.tocShellOpen)}>
      <button
        type="button"
        className={styles.toggleButton}
        onClick={() => setIsOpen((value) => !value)}
        aria-expanded={isOpen}
        aria-controls="doc-right-toc">
        {isOpen ? '收起大纲' : '打开大纲'}
      </button>
      <div
        id="doc-right-toc"
        className={clsx(styles.panel, !isOpen && styles.panelHidden)}
        aria-hidden={!isOpen}>
        <div className={styles.panelHeader}>页面大纲</div>
        <TOC
          toc={toc}
          minHeadingLevel={frontMatter.toc_min_heading_level}
          maxHeadingLevel={frontMatter.toc_max_heading_level}
          className={clsx(ThemeClassNames.docs.docTocDesktop, styles.tocBody)}
        />
        <div className={styles.extraOutline}>
          <button
            type="button"
            className={styles.extraOutlineButton}
            onClick={scrollToComments}>
            评论区
          </button>
        </div>
      </div>
    </div>
  );
}
