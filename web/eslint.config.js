import js from "@eslint/js";
import react from "eslint-plugin-react";
import reactHooks from "eslint-plugin-react-hooks";
import globals from "globals";

export default [
  { ignores: ["dist/"] },
  js.configs.recommended,
  {
    files: ["src/**/*.{js,jsx}"],
    ...react.configs.flat.recommended,
    languageOptions: {
      ...react.configs.flat.recommended.languageOptions,
      globals: globals.browser,
    },
    settings: { react: { version: "detect" } },
  },
  { files: ["src/**/*.{js,jsx}"], ...react.configs.flat["jsx-runtime"] },
  // React 19 no longer checks propTypes, and D1 adds no prop-types package.
  { files: ["src/**/*.{js,jsx}"], rules: { "react/prop-types": "off" } },
  { files: ["src/**/*.{js,jsx}"], ...reactHooks.configs.flat.recommended },
  {
    files: ["*.js", "scripts/**/*.js", "tests/**/*.js"],
    languageOptions: { globals: globals.node },
  },
  {
    files: ["tests/**/*.js"],
    ignores: ["tests/fixtures.js"],
    rules: {
      "no-restricted-imports": [
        "error",
        {
          name: "@playwright/test",
          message: "Import test and expect from ./fixtures.js, which blocks outbound network.",
        },
      ],
    },
  },
];
