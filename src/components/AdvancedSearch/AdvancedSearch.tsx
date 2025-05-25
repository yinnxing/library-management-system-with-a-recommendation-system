import React, { useState, useEffect } from 'react';
import {
  Box,
  TextField,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Pagination,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Grid,
  Typography,
  IconButton,
  Alert,
  CircularProgress,
} from '@mui/material';
import { ArrowUpward, ArrowDownward } from '@mui/icons-material';

interface Book {
  bookId: number;
  title: string;
  author: string;
  publisher: string;
  publicationYear: number;
  isbn: string;
  genre: string;
  descriptions: string;
  coverImageUrl: string;
  quantity: number;
  availableQuantity: number;
  createdAt: string;
  previewLink: string;
}

interface SearchFilters {
  title: string;
  author: string;
  publisher: string;
  genre: string;
  yearFrom: string;
  yearTo: string;
}

const AdvancedSearch: React.FC = () => {
  const [books, setBooks] = useState<Book[]>([]);
  const [filters, setFilters] = useState<SearchFilters>({
    title: '',
    author: '',
    publisher: '',
    genre: '',
    yearFrom: '',
    yearTo: '',
  });
  const [page, setPage] = useState(2);
  const [totalPages, setTotalPages] = useState(1);
  const [sortField, setSortField] = useState<string>('');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchBooks = async () => {
  setLoading(true);
  setError(null);
  try {
    const queryParams = new URLSearchParams();
    queryParams.append('page', (page - 1).toString());
    queryParams.append('size', '5');

    Object.entries(filters).forEach(([key, value]) => {
      if (value) queryParams.append(key, value);
    });

    if (sortField) {
      queryParams.append('sort', `${sortField},${sortDirection}`);
    }

    const response = await fetch(`http://localhost:8080/api/books?${queryParams.toString()}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      credentials: 'include',
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();

    if (data.code === 0) {
      setBooks(data.result.content);
      setTotalPages(data.result.totalPages);
    } else {
      throw new Error(data.message || 'API returned an error');
    }
  } catch (error) {
    console.error('Error fetching books:', error);
    setError(error instanceof Error ? error.message : 'Failed to fetch books');
    setBooks([]);
    setTotalPages(1);
  } finally {
    setLoading(false);
  }
};


  useEffect(() => {
    fetchBooks();
  }, [page, sortField, sortDirection]);

  const handleFilterChange = (field: keyof SearchFilters) => (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setFilters((prev) => ({
      ...prev,
      [field]: event.target.value,
    }));
  };

  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  const handleSearch = () => {
    setPage(1);
    fetchBooks();
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Advanced Book Search
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Search Filters */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              fullWidth
              label="Title"
              value={filters.title}
              onChange={handleFilterChange('title')}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              fullWidth
              label="Author"
              value={filters.author}
              onChange={handleFilterChange('author')}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              fullWidth
              label="Publisher"
              value={filters.publisher}
              onChange={handleFilterChange('publisher')}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              fullWidth
              label="Genre"
              value={filters.genre}
              onChange={handleFilterChange('genre')}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              fullWidth
              label="Year From"
              type="number"
              value={filters.yearFrom}
              onChange={handleFilterChange('yearFrom')}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              fullWidth
              label="Year To"
              type="number"
              value={filters.yearTo}
              onChange={handleFilterChange('yearTo')}
            />
          </Grid>
          <Grid item xs={12}>
            <Button
              variant="contained"
              color="primary"
              onClick={handleSearch}
              sx={{ mt: 2 }}
            >
              Search
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Results Table */}
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
          <CircularProgress />
        </Box>
      ) : (
        <>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>
                    Title
                    <IconButton
                      size="small"
                      onClick={() => handleSort('title')}
                    >
                      {sortField === 'title' ? (
                        sortDirection === 'asc' ? <ArrowUpward /> : <ArrowDownward />
                      ) : null}
                    </IconButton>
                  </TableCell>
                  <TableCell>
                    Author
                    <IconButton
                      size="small"
                      onClick={() => handleSort('author')}
                    >
                      {sortField === 'author' ? (
                        sortDirection === 'asc' ? <ArrowUpward /> : <ArrowDownward />
                      ) : null}
                    </IconButton>
                  </TableCell>
                  <TableCell>Publisher</TableCell>
                  <TableCell>Year</TableCell>
                  <TableCell>Genre</TableCell>
                  <TableCell>Available</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {books.map((book) => (
                  <TableRow key={book.bookId}>
                    <TableCell>{book.title}</TableCell>
                    <TableCell>{book.author}</TableCell>
                    <TableCell>{book.publisher}</TableCell>
                    <TableCell>{book.publicationYear}</TableCell>
                    <TableCell>{book.genre}</TableCell>
                    <TableCell>{book.availableQuantity}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>

          {/* Pagination */}
          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
            <Pagination
              count={totalPages}
              page={page}
              onChange={(_, value) => setPage(value)}
              color="primary"
            />
          </Box>
        </>
      )}
    </Box>
  );
};

export default AdvancedSearch; 