import React, {type ReactNode} from 'react';
import DocCategoryGeneratedIndexPage from '@theme-original/DocCategoryGeneratedIndexPage';
import type DocCategoryGeneratedIndexPageType from '@theme/DocCategoryGeneratedIndexPage';
import type {WrapperProps} from '@docusaurus/types';
import Comment from '@site/src/components/Comment';

type Props = WrapperProps<typeof DocCategoryGeneratedIndexPageType>;

export default function DocCategoryGeneratedIndexPageWrapper(props: Props): ReactNode {
  return (
    <>
      <DocCategoryGeneratedIndexPage {...props} />
      <Comment />
    </>
  );
}
