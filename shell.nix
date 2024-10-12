let
  pkgs = import <nixpkgs> {};
  python = pkgs.python312;
  pythonPackages = python.pkgs;
  lib-path = with pkgs; lib.makeLibraryPath [
    libffi
    openssl
    stdenv.cc.cc
  ];
in with pkgs; mkShell {
  packages = [

    pythonPackages.matplotlib
    pythonPackages.numpy
    pythonPackages.autograd
    pythonPackages.scikit-learn
    pythonPackages.venvShellHook
    # pythonPackages.ipykernel

  ];

  buildInputs = [
    readline
    libffi
    openssl
    git
    openssh
    rsync
    pkg-config
    zlib
    uv
  ];

  shellHook = ''
    SOURCE_DATE_EPOCH=$(date +%s)
    export "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${lib-path}"
    VENV=.venv

    if test ! -d $VENV; then
      python3.12 -m venv $VENV
    fi

    source ./$VENV/bin/activate
    uv pip install -r requirements.txt
  '';

  postShellHook = ''
    ln -sf ${python.sitePackages}/* ./.venv/lib/python3.12/site-packages
  '';
}

