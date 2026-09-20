# Release archives

The release ZIP files in this directory are stored with Git Large File Storage
(Git LFS). Install Git LFS before cloning, or retrieve the archives in an existing
checkout with:

```shell
git lfs install
git lfs pull
```

Git stores small pointer files; Git LFS stores the original ZIP bytes. New ZIP
archives in this directory are automatically tracked by the repository's
`.gitattributes` rule. Keep that rule when adding future releases.

## Importing a new version

Keep each supplied release ZIP here and copy the contents of its top-level
`facemath_wave_XX` directory into the repository root. Preserve the repository's
Git metadata, `.gitattributes`, and this archive guidance when importing.

Commit the related source, tests, documentation, evidence, and release ZIP as one
version update. Thousands of evidence files do not by themselves require separate
commits. Git records the changed content, and the ZIP is uploaded through Git LFS
automatically during a normal push.
