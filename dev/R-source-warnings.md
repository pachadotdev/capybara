util.c:1745:42: warning: returning 'const char *' from a function with result type 'char *' discards
      qualifiers [-Wincompatible-pointer-types-discards-qualifiers]
 1745 |     if(!mbcslocale || utf8locale) return strchr(s, c);
      |                                          ^~~~~~~~~~~~
/usr/include/string.h:265:3: note: expanded from macro 'strchr'
  265 |   __glibc_const_generic (S, const char *, strchr (S, C))
      |   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/usr/include/sys/cdefs.h:838:3: note: expanded from macro '__glibc_const_generic'
  838 |   _Generic (0 ? (PTR) : (void *) 1,                     \
      |   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  839 |             const void *: (CTYPE) (CALL),               \
      |             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  840 |             default: CALL)
      |             ~~~~~~~~~~~~~~
util.c:1760:42: warning: returning 'const char *' from a function with result type 'char *' discards
      qualifiers [-Wincompatible-pointer-types-discards-qualifiers]
 1760 |     if(!mbcslocale || utf8locale) return strrchr(s, c);
      |                                          ^~~~~~~~~~~~~
/usr/include/string.h:296:3: note: expanded from macro 'strrchr'
  296 |   __glibc_const_generic (S, const char *, strrchr (S, C))
      |   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/usr/include/sys/cdefs.h:838:3: note: expanded from macro '__glibc_const_generic'
  838 |   _Generic (0 ? (PTR) : (void *) 1,                     \
      |   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  839 |             const void *: (CTYPE) (CALL),               \
      |             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  840 |             default: CALL)
      |             ~~~~~~~~~~~~~~