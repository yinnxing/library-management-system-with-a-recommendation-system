import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import '../../styles/design-system.css';
import './AdvancedSearch.css';

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
  publicationYearStart: string;
  publicationYearEnd: string;
  availabilityFilter: string;
}

const AdvancedSearch: React.FC = () => {
  const [books, setBooks] = useState<Book[]>([]);
  const [filters, setFilters] = useState<SearchFilters>({
    title: '',
    author: '',
    publisher: '',
    genre: '',
    publicationYearStart: '',
    publicationYearEnd: '',
    availabilityFilter: 'all',
  });
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalResults, setTotalResults] = useState(0);
  const [sortField, setSortField] = useState<string>('title');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');

  const genres = [
    'Fiction', 'Non-Fiction', 'Science Fiction', 'Fantasy', 'Mystery', 
    'Romance', 'Thriller', 'Biography', 'History', 'Science', 
    'Technology', 'Self-Help', 'Business', 'Art', 'Philosophy'
  ];

  const fetchBooks = async () => {
    setLoading(true);
    setError(null);
    try {
      const queryParams = new URLSearchParams();
      queryParams.append('page', page.toString()); // API uses 0-based indexing
      queryParams.append('size', '12');

      // Add filters
      Object.entries(filters).forEach(([key, value]) => {
        if (value && key !== 'availabilityFilter') {
          queryParams.append(key, value);
        }
      });

      // Add sorting
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
        let filteredBooks = data.result.content;
        
        // Apply availability filter on frontend
        if (filters.availabilityFilter === 'available') {
          filteredBooks = filteredBooks.filter((book: Book) => book.availableQuantity > 0);
        } else if (filters.availabilityFilter === 'unavailable') {
          filteredBooks = filteredBooks.filter((book: Book) => book.availableQuantity === 0);
        }

        setBooks(filteredBooks);
        setTotalPages(data.result.totalPages);
        setTotalResults(data.result.totalElements);
        setHasSearched(true);
      } else {
        throw new Error(data.message || 'API returned an error');
      }
    } catch (error) {
      console.error('Error fetching books:', error);
      setError(error instanceof Error ? error.message : 'Failed to fetch books');
      setBooks([]);
      setTotalPages(1);
      setTotalResults(0);
    } finally {
      setLoading(false);
    }
  };

  const handleFilterChange = (field: keyof SearchFilters) => (
    event: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
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
    setPage(1);
  };

  const handleSearch = () => {
    setPage(1);
    fetchBooks();
  };

  const handleClearFilters = () => {
    setFilters({
      title: '',
      author: '',
      publisher: '',
      genre: '',
      publicationYearStart: '',
      publicationYearEnd: '',
      availabilityFilter: 'all',
    });
    setPage(1);
    setSortField('title');
    setSortDirection('asc');
  };

  const handlePageChange = (newPage: number) => {
    setPage(newPage);
    fetchBooks();
  };

  useEffect(() => {
    if (hasSearched) {
      fetchBooks();
    }
  }, [page, sortField, sortDirection]);

  const renderPagination = () => {
    if (totalPages <= 1) return null;

    const pages: JSX.Element[] = [];
    const maxVisiblePages = 5;
    let startPage = Math.max(1, page - Math.floor(maxVisiblePages / 2));
    let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);

    if (endPage - startPage + 1 < maxVisiblePages) {
      startPage = Math.max(1, endPage - maxVisiblePages + 1);
    }

    // Previous button
    if (page > 1) {
      pages.push(
        <button
          key="prev"
          onClick={() => handlePageChange(page - 1)}
          className="pagination-button"
        >
          ‹
        </button>
      );
    }

    // Page numbers
    for (let i = startPage; i <= endPage; i++) {
      pages.push(
        <button
          key={i}
          onClick={() => handlePageChange(i)}
          className={`pagination-button ${i === page ? 'active' : ''}`}
        >
          {i}
        </button>
      );
    }

    // Next button
    if (page < totalPages) {
      pages.push(
        <button
          key="next"
          onClick={() => handlePageChange(page + 1)}
          className="pagination-button"
        >
          ›
        </button>
      );
    }

    return <div className="pagination">{pages}</div>;
  };

  return (
    <div className="advanced-search-container">
      {/* Header */}
      <div className="search-header">
        <div className="header-content">
          <h1 className="page-title">
            <span className="title-icon">🔍</span>
            Tìm Kiếm Nâng Cao
          </h1>
          <p className="page-subtitle">
            Tìm kiếm sách với nhiều tiêu chí và bộ lọc chi tiết
          </p>
        </div>
      </div>

      {/* Search Filters */}
      <div className="search-filters">
        <div className="filters-header">
          <h2 className="filters-title">
            <span className="filters-icon">⚙️</span>
            Bộ Lọc Tìm Kiếm
          </h2>
        </div>

        <div className="filters-grid">
          <div className="filter-group">
            <label className="filter-label">Tên sách</label>
            <input
              type="text"
              className="filter-input"
              placeholder="Nhập tên sách..."
              value={filters.title}
              onChange={handleFilterChange('title')}
            />
          </div>

          <div className="filter-group">
            <label className="filter-label">Tác giả</label>
            <input
              type="text"
              className="filter-input"
              placeholder="Nhập tên tác giả..."
              value={filters.author}
              onChange={handleFilterChange('author')}
            />
          </div>

          <div className="filter-group">
            <label className="filter-label">Nhà xuất bản</label>
            <input
              type="text"
              className="filter-input"
              placeholder="Nhập nhà xuất bản..."
              value={filters.publisher}
              onChange={handleFilterChange('publisher')}
            />
          </div>

          <div className="filter-group">
            <label className="filter-label">Thể loại</label>
            <select
              className="filter-select"
              value={filters.genre}
              onChange={handleFilterChange('genre')}
            >
              <option value="">Tất cả thể loại</option>
              {genres.map(genre => (
                <option key={genre} value={genre}>{genre}</option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label className="filter-label">Năm xuất bản từ</label>
            <input
              type="number"
              className="filter-input"
              placeholder="VD: 2000"
              min="1900"
              max={new Date().getFullYear()}
              value={filters.publicationYearStart}
              onChange={handleFilterChange('publicationYearStart')}
            />
          </div>

          <div className="filter-group">
            <label className="filter-label">Năm xuất bản đến</label>
            <input
              type="number"
              className="filter-input"
              placeholder="VD: 2024"
              min="1900"
              max={new Date().getFullYear()}
              value={filters.publicationYearEnd}
              onChange={handleFilterChange('publicationYearEnd')}
            />
          </div>

          <div className="filter-group">
            <label className="filter-label">Tình trạng</label>
            <select
              className="filter-select"
              value={filters.availabilityFilter}
              onChange={handleFilterChange('availabilityFilter')}
            >
              <option value="all">Tất cả</option>
              <option value="available">Còn sách</option>
              <option value="unavailable">Hết sách</option>
            </select>
          </div>
        </div>

        <div className="filters-actions">
          <button
            className="search-button primary"
            onClick={handleSearch}
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="loading-spinner"></span>
                Đang tìm...
              </>
            ) : (
              <>
                <span className="button-icon">🔍</span>
                Tìm kiếm
              </>
            )}
          </button>
          <button
            className="search-button secondary"
            onClick={handleClearFilters}
          >
            <span className="button-icon">🗑️</span>
            Xóa bộ lọc
          </button>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="error-message">
          <span className="error-icon">⚠️</span>
          {error}
        </div>
      )}

      {/* Results Section */}
      {hasSearched && (
        <div className="search-results">
          <div className="results-header">
            <div className="results-info">
              <h3 className="results-title">
                Kết quả tìm kiếm
                {totalResults > 0 && (
                  <span className="results-count">({totalResults} kết quả)</span>
                )}
              </h3>
            </div>

            {books.length > 0 && (
              <div className="results-controls">
                <div className="view-toggle">
                  <button
                    className={`view-button ${viewMode === 'grid' ? 'active' : ''}`}
                    onClick={() => setViewMode('grid')}
                  >
                    ⊞
                  </button>
                  <button
                    className={`view-button ${viewMode === 'list' ? 'active' : ''}`}
                    onClick={() => setViewMode('list')}
                  >
                    ☰
                  </button>
                </div>

                <div className="sort-controls">
                  <label className="sort-label">Sắp xếp:</label>
                  <select
                    className="sort-select"
                    value={`${sortField}-${sortDirection}`}
                    onChange={(e) => {
                      const [field, direction] = e.target.value.split('-');
                      setSortField(field);
                      setSortDirection(direction as 'asc' | 'desc');
                    }}
                  >
                    <option value="title-asc">Tên A-Z</option>
                    <option value="title-desc">Tên Z-A</option>
                    <option value="author-asc">Tác giả A-Z</option>
                    <option value="author-desc">Tác giả Z-A</option>
                    <option value="publicationYear-desc">Năm mới nhất</option>
                    <option value="publicationYear-asc">Năm cũ nhất</option>
                  </select>
                </div>
              </div>
            )}
          </div>

          {loading ? (
            <div className="loading-container">
              <div className="loading-spinner large"></div>
              <p className="loading-text">Đang tìm kiếm sách...</p>
            </div>
          ) : books.length > 0 ? (
            <>
              <div className={`books-container ${viewMode}`}>
                {books.map((book) => (
                  <div key={book.bookId} className="book-item">
                    <div className="book-image">
                      <img
                        src={book.coverImageUrl}
                        alt={book.title}
                        onError={(e) => {
                          (e.target as HTMLImageElement).src = '/placeholder-book.jpg';
                        }}
                      />
                      <div className="book-overlay">
                        <Link to={`/books/${book.bookId}`} className="view-details-btn">
                          Xem chi tiết
                        </Link>
                      </div>
                    </div>
                    
                    <div className="book-info">
                      <h4 className="book-title">
                        <Link to={`/books/${book.bookId}`}>{book.title}</Link>
                      </h4>
                      <p className="book-author">
                        <span className="label">Tác giả:</span>
                        <span className="value">{book.author}</span>
                      </p>
                      <p className="book-publisher">
                        <span className="label">NXB:</span>
                        <span className="value">{book.publisher}</span>
                      </p>
                      <p className="book-year">
                        <span className="label">Năm:</span>
                        <span className="value">{book.publicationYear}</span>
                      </p>
                      <p className="book-genre">
                        <span className="label">Thể loại:</span>
                        <span className="value">{book.genre}</span>
                      </p>
                      <div className="book-availability">
                        <span className="label">Tình trạng:</span>
                        <span className={`availability-badge ${book.availableQuantity > 0 ? 'available' : 'unavailable'}`}>
                          {book.availableQuantity > 0 ? `Còn ${book.availableQuantity} cuốn` : 'Hết sách'}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {renderPagination()}
            </>
          ) : (
            <div className="empty-results">
              <div className="empty-icon">📚</div>
              <h3 className="empty-title">Không tìm thấy sách nào</h3>
              <p className="empty-description">
                Thử điều chỉnh bộ lọc tìm kiếm hoặc sử dụng từ khóa khác
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default AdvancedSearch;
