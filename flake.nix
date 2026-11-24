{
  description = "ilo-toki-lili: small monolingual Toki Pona language model";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs:
        let
          pythonEnv = pkgs.python3.withPackages (ps: with ps; [
            torch
            transformers
            datasets
            tqdm
          ]);
        in
        {
          default = pkgs.mkShell {
            packages = [ pythonEnv ];
            shellHook = ''
              # Stable symlink for editors, since store paths change on updates.
              ln -sfn "${pythonEnv}" .python-env
            '';
          };
        });
    };
}
