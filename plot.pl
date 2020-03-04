#!/usr/bin/perl -s

use Chart::Gnuplot;
use File::Temp qw(tempfile);

my $k = $ARGV[0] || 'genrms';
my @t;
my @v;

while (<STDIN>) {
  chomp;
  my %kv = map { split /=/, $_, 2 } split /\s+/;
  my $v = $kv{$k};
  defined($v) or next;
  my $t = $kv{genrounds};
  defined($t) or next;
  push @v, $v;
  push @t, $t;
}

my ($fh, $fn) = tempfile();

my $chart = Chart::Gnuplot->new(
  output => $fn,
  xlabel => "t",
  ylabel => $k,
  xrange => [$x0, $x1],
  yrange => [$y0, $y1],
  $log ? (logscale => 'x') : ( ),
  terminal => "png",
);

my $dataset = Chart::Gnuplot::DataSet->new(
  xdata => \@t,
  ydata => \@v,
  style => 'lines',
);

$chart->plot2d($dataset);

my $png;
{
  undef local $/;
  $png = <$fh>;
  close($fh);
};

print $png
