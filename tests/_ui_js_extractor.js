
function isIdentifierChar(ch) {
  return /[A-Za-z0-9_$]/.test(ch || '');
}
function previousSignificantToken(source, index) {
  let i = index - 1;
  while (i >= 0 && /\s/.test(source[i])) i -= 1;
  if (i < 0) return '';
  if (!isIdentifierChar(source[i])) return source[i];
  const end = i + 1;
  while (i >= 0 && isIdentifierChar(source[i])) i -= 1;
  return source.slice(i + 1, end);
}
function canStartRegexLiteral(source, index) {
  const token = previousSignificantToken(source, index);
  if (!token) return true;
  if ('({[=,:;!&|?+-*~^<>'.includes(token)) return true;
  return ['return','throw','case','delete','typeof','void','new','in','of','yield','await'].includes(token);
}
function skipQuotedLiteral(source, index, quote) {
  for (let i = index + 1; i < source.length; i += 1) {
    if (source[i] === '\\') { i += 1; continue; }
    if (source[i] === quote) return i;
  }
  throw new Error('unterminated string literal');
}
function skipTemplateLiteral(source, index) {
  for (let i = index + 1; i < source.length; i += 1) {
    if (source[i] === '\\') { i += 1; continue; }
    if (source[i] === '`') return i;
    if (source[i] === '$' && source[i + 1] === '{') {
      let depth = 1;
      i += 1;
      while (i < source.length && depth > 0) {
        i += 1;
        if (source[i] === '{') depth += 1;
        else if (source[i] === '}') depth -= 1;
      }
    }
  }
  throw new Error('unterminated template literal');
}
function skipLineComment(source, index) {
  const nl = source.indexOf('\n', index);
  return nl < 0 ? source.length - 1 : nl;
}
function skipBlockComment(source, index) {
  const end = source.indexOf('*/', index + 2);
  if (end < 0) throw new Error('unterminated block comment');
  return end + 1;
}
function skipRegexLiteral(source, index) {
  let inClass = false;
  for (let i = index + 1; i < source.length; i += 1) {
    const ch = source[i];
    const next = source[i + 1];
    if (ch === '\\') { i += 1; continue; }
    if (ch === '[') inClass = true;
    else if (ch === ']') inClass = false;
    else if (ch === '/' && !inClass) {
      while (/[A-Za-z]/.test(source[i + 1] || '')) i += 1;
      return i;
    }
  }
  throw new Error('unterminated regex literal');
}
function extractFunction(source, name) {
  const marker = 'function ' + name + '(';
  const start = source.indexOf(marker);
  if (start < 0) throw new Error('not found: ' + name);
  const brace = source.indexOf('{', source.indexOf(')', start));
  let depth = 0;
  for (let i = brace; i < source.length; i += 1) {
    const ch = source[i];
    const next = source[i + 1];
    if (ch === '"' || ch === "'") i = skipQuotedLiteral(source, i, ch);
    else if (ch === '`') i = skipTemplateLiteral(source, i);
    else if (ch === '/' && next === '/') i = skipLineComment(source, i);
    else if (ch === '/' && next === '*') i = skipBlockComment(source, i);
    else if (ch === '/' && canStartRegexLiteral(source, i)) i = skipRegexLiteral(source, i);
    else if (ch === '{') depth += 1;
    else if (ch === '}') { depth -= 1; if (depth === 0) return source.slice(start, i + 1); }
  }
  throw new Error('unterminated: ' + name);
}
