function getText(node) {
  if (!node) {
    return '';
  }

  if (node.type === 'text' || node.type === 'inlineCode') {
    return node.value ?? '';
  }

  if (!Array.isArray(node.children)) {
    return '';
  }

  return node.children.map(getText).join('');
}

function normalizeWhitespace(value) {
  return value.replace(/\s+/g, ' ').trim();
}

function parseTags(rawValue) {
  const matches = rawValue.match(/\[([^\]]+)\]/g) ?? [];
  return matches
    .map((item) => item.slice(1, -1).trim())
    .filter(Boolean);
}

function parseMetaItem(listItem) {
  const text = normalizeWhitespace(getText(listItem));
  const match = text.match(/^(tags|authors|institution|link|Simple LLM Summary):\s*(.+)$/i);

  if (!match) {
    return null;
  }

  return {
    key: match[1].toLowerCase(),
    value: match[2].trim(),
  };
}

function parsePaperItem(listItem) {
  if (!Array.isArray(listItem.children) || listItem.children.length < 2) {
    return null;
  }

  const [titleNode, metaList] = listItem.children;
  if (titleNode?.type !== 'paragraph' || metaList?.type !== 'list') {
    return null;
  }

  const title = normalizeWhitespace(getText(titleNode));
  if (!title) {
    return null;
  }

  const paper = {
    title,
    authors: '',
    institution: '',
    link: '',
    summary: '',
    tags: [],
  };

  for (const metaItem of metaList.children ?? []) {
    const parsed = parseMetaItem(metaItem);
    if (!parsed) {
      continue;
    }

    if (parsed.key === 'tags') {
      paper.tags = parseTags(parsed.value);
    } else if (parsed.key === 'authors') {
      paper.authors = parsed.value;
    } else if (parsed.key === 'institution') {
      paper.institution = parsed.value;
    } else if (parsed.key === 'link') {
      paper.link = parsed.value;
    } else if (parsed.key === 'simple llm summary') {
      paper.summary = parsed.value;
    }
  }

  if (!paper.link || !paper.authors || !paper.institution) {
    return null;
  }

  return paper;
}

function createAttribute(name, value) {
  return {
    type: 'mdxJsxAttribute',
    name,
    value,
  };
}

function createPaperCardNode(paper) {
  return {
    type: 'mdxJsxFlowElement',
    name: 'PaperCard',
    attributes: [
      createAttribute('title', paper.title),
      createAttribute('link', paper.link),
      createAttribute('authors', paper.authors),
      createAttribute('institution', paper.institution),
      createAttribute('summary', paper.summary),
      createAttribute('tags', JSON.stringify(paper.tags)),
    ],
    children: [],
  };
}

function createSectionStatCardNode(text) {
  return {
    type: 'mdxJsxFlowElement',
    name: 'SectionStatCard',
    attributes: [createAttribute('text', text)],
    children: [],
  };
}

function parseSectionStat(node) {
  if (node?.type !== 'paragraph') {
    return null;
  }

  const text = normalizeWhitespace(getText(node));
  if (!text) {
    return null;
  }

  const statPattern = /total:\s*\d+$/i;
  if (!statPattern.test(text)) {
    return null;
  }

  if (
    text.startsWith('cs.DC total:') ||
    text.startsWith('cs.AI/cs.LG contains')
  ) {
    return text;
  }

  return null;
}

function transformChildren(node) {
  if (!Array.isArray(node.children)) {
    return;
  }

  const nextChildren = [];

  for (const child of node.children) {
    const sectionStat = parseSectionStat(child);
    if (sectionStat) {
      nextChildren.push(createSectionStatCardNode(sectionStat));
      continue;
    }

    if (child.type === 'list') {
      const papers = child.children.map(parsePaperItem);
      const isPaperList =
        papers.length > 0 &&
        papers.every((paper) => paper !== null);

      if (isPaperList) {
        nextChildren.push(...papers.map(createPaperCardNode));
        continue;
      }
    }

    transformChildren(child);
    nextChildren.push(child);
  }

  node.children = nextChildren;
}

export default function remarkPaperCards() {
  return (tree) => {
    transformChildren(tree);
  };
}
