#!/usr/bin/perl

$| = 1;
@ARGV or die "Usage: $0 pic.any\n";
my $fn = shift(@ARGV);

if (my $filter = $ENV{TEWEL_PICREADER_FILTER}) {
  open(STDOUT, "| $filter") or die "$0: $filter: $!";
}

select(STDOUT);
$| = 1;

exec('convert', $fn, "-depth", '8', '+set', 'comment', 'ppm:-');
die "$0: convert: $!\n";
