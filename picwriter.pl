#!/usr/bin/perl

@ARGV or die "Usage: $0 pic.any\n";
my $fn = shift(@ARGV);

if (my $filter = $ENV{TEWEL_PICWRITER_FILTER}) {
  $filter =~ s#^//#/opt/makemore/share/tewel/#;
  open(STDIN, "$filter |") or die "$0: $filter: $!";
}

my ($ext) = ($fn =~ /\.(.+$)/);
$ext ||= 'ppm';

system('convert', 'ppm:-', "$ext:$fn.tmp");
if ($? == -1) {
  die "$0: convert: $!\n";
} elsif ($? & 127) {
  my $sig = $? & 127;
  die "$0: convert: signal $sig\n";
} elsif ($? >> 8) {
  my $ret = $? >> 8;
  die "$0: convert: ret $ret\n";
}

$ret = rename("$fn.tmp", $fn);
$ret or die "$0: rename: $!";

exit 0;
