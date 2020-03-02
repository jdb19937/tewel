#!/usr/bin/perl -s

use Chart::Gnuplot;
use File::Temp qw(tempfile);

my $k = $ARGV[0] || 'genrms';
my @v;

while (<STDIN>) {
  chomp;
  my %kv = map { split /=/, $_, 2 } split /\s+/;
  my $v = $kv{$k};
  defined($v) or next;
  push @v, $v;
}

my ($fh, $fn) = tempfile();

my $chart = Chart::Gnuplot->new(
  output => $fn,
  xlabel => "t",
  ylabel => $k,
  yrange => [$y0, $y1],
  terminal => "png",
);

my $dataset = Chart::Gnuplot::DataSet->new(
  xdata => [0 .. $#v],
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
