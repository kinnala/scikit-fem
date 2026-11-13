"""Sphinx directive for embedding interactive scikit-fem plots.

The content of the directive is Python code whose last line is an expression
evaluating to :class:`skfem.visuals.web.WebPlot`::

    .. skfem-web::
       :height: 400
       :caption: The solution of Example 1.

       from docs.examples.ex01 import visualize
       visualize()

The page contains only a placeholder.  The data of each plot is written to
its own file under ``_static/skfem-web`` and a small loader script creates
the viewer when the placeholder scrolls into view.  At most ``MAX_LIVE``
viewers are kept alive: browsers limit the number of WebGL contexts per page.
Both scripts are loaded only on the pages that use the directive.  Builders
other than HTML skip the plot.

"""

import ast
import hashlib
import os
import sys

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective

from skfem.visuals.web import WebPlot, viewer_js


VIEWER = 'skfem-viewer.js'
LOADER = 'skfem-web.js'
DATADIR = 'skfem-web'
MARKER = 'skfem-web'
MAX_LIVE = 8

# The data files are scripts and not JSON so that the documentation works
# also when opened directly from the disk: fetch is blocked on file://.
LOADER_JS = """
(function () {
  'use strict';
  var MAX_LIVE = %d;
  var base = document.currentScript.src.replace(/[^\\/]*$/, '') + '%s/';
  var data = {}, waiting = {}, live = [];
  window.SkfemWeb = {
    data: data,
    register: function (key, payload) { data[key] = payload; },
  };

  function load(key, done) {
    if (data[key]) return done();
    if (waiting[key]) return waiting[key].push(done);
    waiting[key] = [done];
    var s = document.createElement('script');
    s.src = base + key + '.js';
    s.onload = function () {
      var fns = waiting[key];
      delete waiting[key];
      fns.forEach(function (f) { f(); });
    };
    s.onerror = function () {
      delete waiting[key];
      console.error('skfem-web: cannot load ' + s.src);
    };
    document.head.appendChild(s);
  }

  function unmount(el) {
    el.skfemPlot.destroy();
    el.skfemPlot = null;
    live.splice(live.indexOf(el), 1);
  }

  // keep at most MAX_LIVE viewers; destroy the ones that were on the
  // screen the longest time ago, never the ones that are visible now
  function evict() {
    var hidden = live.filter(function (el) { return !el.skfemVisible; });
    hidden.sort(function (a, b) { return a.skfemSeen - b.skfemSeen; });
    while (live.length > MAX_LIVE && hidden.length) unmount(hidden.shift());
  }

  function mount(el) {
    if (el.skfemPlot) return;
    var key = el.getAttribute('data-skfem');
    load(key, function () {
      if (el.skfemPlot || !el.skfemVisible || !data[key]) return;
      el.skfemPlot = SkfemViewer.mount(el, data[key].data,
                                       data[key].options);
      live.push(el);
      evict();
    });
  }

  function init() {
    var els = document.querySelectorAll('[data-skfem]');
    if (!('IntersectionObserver' in window)) {
      els.forEach(function (el) { el.skfemVisible = true; mount(el); });
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        e.target.skfemVisible = e.isIntersecting;
        e.target.skfemSeen = Date.now();
        if (e.isIntersecting) mount(e.target);
      });
    }, {rootMargin: '300px 0px'});
    els.forEach(function (el) { io.observe(el); });
  }

  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', init);
  else init();
})();
""" % (MAX_LIVE, DATADIR)


def script_safe(s):
    # line and paragraph separators are valid in JSON but not in old JS
    return s.replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')


def datafile(app, key):
    return os.path.join(app.outdir, '_static', DATADIR, key + '.js')


def imported_files(code):
    """The source files of the modules imported by the code."""
    names = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    files = [getattr(sys.modules.get(name), '__file__', None)
             for name in names]
    return [f for f in files if f]


def run_code(code, filename):
    """Execute the code and return the value of the last expression."""
    tree = ast.parse(code, filename)
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        raise ValueError("the last line must be an expression")
    last = ast.Expression(tree.body.pop().value)
    ns = {'__name__': '__skfem_web__'}
    exec(compile(tree, filename, 'exec'), ns)
    return eval(compile(last, filename, 'eval'), ns)


class SkfemWeb(SphinxDirective):

    has_content = True
    option_spec = {
        'height': directives.positive_int,
        'caption': directives.unchanged,
    }

    def run(self):
        source, line = self.get_source_info()
        code = '\n'.join(self.content)
        try:
            plot = run_code(code, '{}:{}'.format(source, line))
            if not isinstance(plot, WebPlot):
                raise TypeError("the last line must evaluate to WebPlot, "
                                "got {}".format(type(plot).__name__))
        except Exception as e:
            return [self.state.document.reporter.warning(
                'skfem-web: {}: {}'.format(type(e).__name__, e), line=line)]
        if 'height' in self.options:
            plot.height = self.options['height']
        # the documentation does not have a dark theme
        plot.options.setdefault('theme', 'light')
        # rebuild the page when the example changes
        for filename in imported_files(code):
            self.env.note_dependency(filename)
        if self.env.app.builder.format != 'html':
            return []
        # the file is named after its content: unchanged plots are not
        # rewritten and browsers can cache them
        payload = script_safe(plot.to_json())
        key = hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]
        filename = datafile(self.env.app, key)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        keys(self.env).setdefault(self.env.docname, set()).add(key)
        if not os.path.exists(filename):
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('SkfemWeb.register("{}",{});\n'.format(key, payload))
        raw = nodes.raw(
            '',
            '<div class="{}" data-skfem="{}" style="width:100%;height:{}px">'
            '</div>'.format(MARKER, key, plot.height),
            format='html', classes=[MARKER])
        if 'caption' not in self.options:
            return [raw]
        figure = nodes.figure('', raw)
        text, messages = self.state.inline_text(self.options['caption'],
                                                self.lineno)
        figure += nodes.caption('', '', *text)
        return [figure] + messages


def keys(env):
    """Document name to the keys of its plots."""
    if not hasattr(env, 'skfem_web_keys'):
        env.skfem_web_keys = {}
    return env.skfem_web_keys


def purge_keys(app, env, docname):
    keys(env).pop(docname, None)


def merge_keys(app, env, docnames, other):
    for docname in docnames:
        if docname in keys(other):
            keys(env)[docname] = keys(other)[docname]


def missing_data(app, env, added, changed, removed):
    # Sphinx does not rerun the directive on unchanged pages: read again
    # the pages whose data files were deleted from the output directory
    if app.builder.format != 'html':
        return []
    return [docname for docname, ks in keys(env).items()
            if not all(os.path.exists(datafile(app, k)) for k in ks)]


def copy_scripts(app):
    if app.builder.format != 'html':
        return
    static = os.path.join(app.outdir, '_static')
    os.makedirs(static, exist_ok=True)
    for name, source in [(VIEWER, viewer_js()), (LOADER, LOADER_JS)]:
        with open(os.path.join(static, name), 'w', encoding='utf-8') as f:
            f.write(source)


def add_viewer(app, pagename, templatename, context, doctree):
    # load the scripts only on the pages that have plots
    if doctree is not None and any(MARKER in node['classes']
                                   for node in doctree.findall(nodes.raw)):
        app.add_js_file(VIEWER)
        app.add_js_file(LOADER)


def setup(app):
    app.add_directive('skfem-web', SkfemWeb)
    app.connect('builder-inited', copy_scripts)
    app.connect('env-purge-doc', purge_keys)
    app.connect('env-merge-info', merge_keys)
    app.connect('env-get-outdated', missing_data)
    app.connect('html-page-context', add_viewer)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
