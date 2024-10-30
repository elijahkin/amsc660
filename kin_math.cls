% Set up font size, margins, paper size, etc.
\LoadClass[11pt, letterpaper, answers]{exam}
\RequirePackage[margin=1in]{geometry}

% Load all the packages we want
\RequirePackage{amssymb}                      % \mathbb{R}
\RequirePackage{amsmath}                      % \begin{equation*}
\RequirePackage{amsthm}                       % \begin{proof}
\RequirePackage[colorlinks = true,
                urlcolor = magenta]{hyperref} % \url{https://www.google.com/}
\RequirePackage[dvipsnames]{xcolor}           % \textcolor{cyan}{Hello!}
\RequirePackage{enumitem}                     % [label=(\alph*)]
\RequirePackage{pgfplots}                     % \begin{tikzpicture}
\RequirePackage{mathtools}                    % \coloneqq
\RequirePackage{parskip}                      % Nicer paragraph spacing
\RequirePackage{mathrsfs}                     % \mathscr{B}
\RequirePackage{float}                        % Proper figure placement
\RequirePackage{tikz-cd}                      % \begin{tikzcd}
\RequirePackage{listings}                     % \begin{lstlisting}

% Remove italics from block text
\theoremstyle{definition}

% Ensure figures are in the right spot by default
\renewcommand*{\fps@figure}{H}

% Make enumerate use (a) by default
\setlist[enumerate]{label=(\alph*)}

% Make numbering specific to the subsection (i.e. Theorem 1.2.1)
\newtheorem{thm}{Theorem}[subsection]

% Define variants of "Theorem"
\newtheorem{lem}[thm]{Lemma}
\newtheorem{defn}[thm]{Definition}
\newtheorem{eg}[thm]{Example}
\newtheorem{ex}[thm]{Exercise}
\newtheorem{cor}[thm]{Corollary}
\newtheorem{rmk}[thm]{Remark}
\newtheorem{prop}[thm]{Proposition}

% Style the footer
\footer{}{\thepage}{}

% Bibliography styling
\nocite{*}
\bibliographystyle{plain}

% Import Tikz libraries
\usetikzlibrary{arrows}
\usetikzlibrary{calc}
\usetikzlibrary{patterns}
\usetikzlibrary{shapes.geometric}
\usetikzlibrary{decorations.markings}
\usepgfplotslibrary{fillbetween}

% Styling for contours
\pgfplotsset{
    compat=1.18,
    interpret as polar/.style={
            x filter/.code=\pgfmathparse{cos(rawx)*rawy},
            y filter/.code=\pgfmathparse{sin(rawx)*rawy}
        }
}

% Styling for code blocks
\lstset{
  basicstyle=\ttfamily\footnotesize,
  keywordstyle=\color{Magenta},
  commentstyle=\color{Green},
  showstringspaces=false
}

% Leave to-dos in red text
\newcommand{\todo}[1]{\emph{\textcolor{red}{#1}}}
