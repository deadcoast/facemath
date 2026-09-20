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
