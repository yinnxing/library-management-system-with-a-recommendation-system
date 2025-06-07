import React, { useState, useEffect } from 'react';
import styles from './BookManagement.module.css'; 
import BookModal from './BookModal'; 
import AdminApi from '../../../api/AdminApi';

const BookManagement = () => {
  const [books, setBooks] = useState([]);
  const [allBooks, setAllBooks] = useState([]); // For stats calculation
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [currentBook, setCurrentBook] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterGenre, setFilterGenre] = useState('All');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(0);
  const [totalElements, setTotalElements] = useState(0);

  const booksPerPage = 12;

  useEffect(() => {
    fetchAllBooks(); // For stats and genre options
  }, []);

  useEffect(() => {
    fetchBooks();
  }, [currentPage, filterGenre]);

  useEffect(() => {
    // Reset to first page when search term changes
    if (currentPage !== 1) {
      setCurrentPage(1);
    } else {
      fetchBooks();
    }
  }, [searchTerm]);

  const fetchAllBooks = async () => {
    try {
      const response = await AdminApi.getAllBooks();
      if (response.data.code === 0) {
        setAllBooks(response.data.result.content);
      }
    } catch (error) {
      console.error('Error fetching all books:', error);
    }
  };

  const fetchBooks = async () => {
    setLoading(true);
    try {
      const response = await AdminApi.getBooks(
        currentPage,
        booksPerPage,
        filterGenre === 'All' ? null : filterGenre
      );
      
      if (response.data.code === 0) {
        const result = response.data.result;
        let filteredBooks = result.content;
        
        // Apply search filter on frontend since API doesn't support search
        if (searchTerm) {
          filteredBooks = result.content.filter(book => 
            book.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
            book.author.toLowerCase().includes(searchTerm.toLowerCase()) ||
            book.isbn.toLowerCase().includes(searchTerm.toLowerCase())
          );
        }
        
        setBooks(filteredBooks);
        setTotalPages(result.totalPages);
        setTotalElements(result.totalElements);
      } else {
        console.error('Error: API returned non-zero code.');
        setBooks([]);
      }
    } catch (error) {
      console.error('Error fetching books:', error);
      setBooks([]);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = (bookId) => {
    if (window.confirm('Bạn có chắc chắn muốn xóa cuốn sách này?')) {
      AdminApi.deleteBook(bookId)
        .then(() => {
          setBooks((prevBooks) => prevBooks.filter((book) => book.bookId !== bookId));
          setAllBooks((prevBooks) => prevBooks.filter((book) => book.bookId !== bookId));
          alert('Xóa sách thành công!');
          // Refresh current page
          fetchBooks();
          fetchAllBooks();
        })
        .catch((error) => console.error('Error deleting book:', error));
    }
  };

  const handleEdit = (book) => {
    setCurrentBook(book);
    setShowModal(true);
  };

  const handleAdd = () => {
    setCurrentBook(null);
    setShowModal(true);
  };

  const handleModalClose = () => {
    setShowModal(false);
    setCurrentBook(null);
  };

  const handleSave = (updatedBook) => {
    if (updatedBook.bookId) {
      AdminApi.editBook(updatedBook.bookId, updatedBook)
        .then((response) => {
          setBooks((prevBooks) =>
            prevBooks.map((book) =>
              book.bookId === response.data.result.bookId ? response.data.result : book
            )
          );
          setAllBooks((prevBooks) =>
            prevBooks.map((book) =>
              book.bookId === response.data.result.bookId ? response.data.result : book
            )
          );
          handleModalClose();
          alert('Cập nhật sách thành công!');
          fetchBooks();
          fetchAllBooks();
        })
        .catch((error) => {
          console.error('Error updating book:', error);
          alert('Cập nhật sách thất bại. Vui lòng thử lại.');
        });
    } else {
      AdminApi.createBook(updatedBook)
        .then((response) => {
          handleModalClose();
          alert('Thêm sách thành công!');
          // Refresh data
          fetchBooks();
          fetchAllBooks();
          // Go to first page to see the new book
          setCurrentPage(1);
        })
        .catch((error) => {
          console.error('Error adding book:', error);
          alert('Thêm sách thất bại. Vui lòng thử lại.');
        });
    }
  };

  const handleCategoryChange = (genre) => {
    setFilterGenre(genre);
    setCurrentPage(1); // Reset to first page when category changes
    setSearchTerm(''); // Clear search when changing category
  };

  const handlePageChange = (page) => {
    setCurrentPage(page);
    // Scroll to top when page changes
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const getBookStats = () => {
    const totalBooks = allBooks.length;
    const totalQuantity = allBooks.reduce((sum, book) => sum + book.quantity, 0);
    const availableQuantity = allBooks.reduce((sum, book) => sum + book.availableQuantity, 0);
    const borrowedQuantity = totalQuantity - availableQuantity;
    const uniqueGenres = [...new Set(allBooks.map(book => book.genre))].length;
    
    return { totalBooks, totalQuantity, availableQuantity, borrowedQuantity, uniqueGenres };
  };

  const stats = getBookStats();
  const uniqueGenres = [...new Set(allBooks.map(book => book.genre))];

  const renderPagination = () => {
    if (totalPages <= 1) return null;

    const pages = [];
    const maxVisiblePages = 5;
    
    let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2));
    let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);
    
    if (endPage - startPage + 1 < maxVisiblePages) {
      startPage = Math.max(1, endPage - maxVisiblePages + 1);
    }

    // Previous button
    if (currentPage > 1) {
      pages.push(
        <button
          key="prev"
          className={styles.paginationButton}
          onClick={() => handlePageChange(currentPage - 1)}
        >
          ‹ Trước
        </button>
      );
    }

    // First page
    if (startPage > 1) {
      pages.push(
        <button
          key={1}
          className={styles.paginationButton}
          onClick={() => handlePageChange(1)}
        >
          1
        </button>
      );
      if (startPage > 2) {
        pages.push(
          <span key="ellipsis1" className={styles.paginationEllipsis}>
            ...
          </span>
        );
      }
    }

    // Page numbers
    for (let i = startPage; i <= endPage; i++) {
      pages.push(
        <button
          key={i}
          className={`${styles.paginationButton} ${
            currentPage === i ? styles.active : ''
          }`}
          onClick={() => handlePageChange(i)}
        >
          {i}
        </button>
      );
    }

    // Last page
    if (endPage < totalPages) {
      if (endPage < totalPages - 1) {
        pages.push(
          <span key="ellipsis2" className={styles.paginationEllipsis}>
            ...
          </span>
        );
      }
      pages.push(
        <button
          key={totalPages}
          className={styles.paginationButton}
          onClick={() => handlePageChange(totalPages)}
        >
          {totalPages}
        </button>
      );
    }

    // Next button
    if (currentPage < totalPages) {
      pages.push(
        <button
          key="next"
          className={styles.paginationButton}
          onClick={() => handlePageChange(currentPage + 1)}
        >
          Tiếp ›
        </button>
      );
    }

    return pages;
  };

  if (loading) {
    return (
      <div className={styles.loadingContainer}>
        <div className={styles.loadingSpinner}></div>
        <p className={styles.loadingMessage}>Đang tải danh sách sách...</p>
      </div>
    );
  }

  return (
    <div className={styles.bookManagementContainer}>
      {/* Header Section */}
      <div className={styles.header}>
        <div className={styles.titleSection}>
          <h2 className={styles.title}>
            <span className={styles.titleIcon}>📚</span>
            Quản lý sách
          </h2>
          <p className={styles.subtitle}>
            Quản lý thông tin sách, thêm mới, chỉnh sửa và xóa sách trong hệ thống
          </p>
        </div>
        <div className={styles.headerActions}>
          <button className={styles.refreshButton} onClick={() => {
            fetchBooks();
            fetchAllBooks();
          }}>
            <span className={styles.refreshIcon}>🔄</span>
            Làm mới
          </button>
          <button onClick={handleAdd} className={styles.addButton}>
            <span className={styles.addIcon}>➕</span>
            Thêm sách mới
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className={styles.statsContainer}>
        <div className={styles.statCard}>
          <div className={styles.statIcon}>📚</div>
          <div className={styles.statContent}>
            <h3 className={styles.statNumber}>{stats.totalBooks}</h3>
            <p className={styles.statLabel}>Tổng số sách</p>
          </div>
        </div>
        <div className={styles.statCard}>
          <div className={styles.statIcon}>📖</div>
          <div className={styles.statContent}>
            <h3 className={styles.statNumber}>{stats.totalQuantity}</h3>
            <p className={styles.statLabel}>Tổng số lượng</p>
          </div>
        </div>
        <div className={styles.statCard}>
          <div className={styles.statIcon}>✅</div>
          <div className={styles.statContent}>
            <h3 className={styles.statNumber}>{stats.availableQuantity}</h3>
            <p className={styles.statLabel}>Có sẵn</p>
          </div>
        </div>
        <div className={styles.statCard}>
          <div className={styles.statIcon}>📋</div>
          <div className={styles.statContent}>
            <h3 className={styles.statNumber}>{stats.borrowedQuantity}</h3>
            <p className={styles.statLabel}>Đang mượn</p>
          </div>
        </div>
        <div className={styles.statCard}>
          <div className={styles.statIcon}>🏷️</div>
          <div className={styles.statContent}>
            <h3 className={styles.statNumber}>{stats.uniqueGenres}</h3>
            <p className={styles.statLabel}>Thể loại</p>
          </div>
        </div>
      </div>

      {/* Filters Section */}
      <div className={styles.filtersContainer}>
        <div className={styles.searchContainer}>
          <span className={styles.searchIcon}>🔍</span>
          <input
            type="text"
            placeholder="Tìm kiếm theo tên sách, tác giả hoặc ISBN..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className={styles.searchInput}
          />
        </div>
        <div className={styles.filterContainer}>
          <label className={styles.filterLabel}>Thể loại:</label>
          <select
            value={filterGenre}
            onChange={(e) => handleCategoryChange(e.target.value)}
            className={styles.filterSelect}
          >
            <option value="All">Tất cả</option>
            {uniqueGenres.map(genre => (
              <option key={genre} value={genre}>{genre}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Results Info */}
      <div className={styles.resultsInfo}>
        <span className={styles.resultsCount}>
          {searchTerm ? (
            `Hiển thị ${books.length} kết quả tìm kiếm từ ${totalElements} cuốn sách`
          ) : (
            `Hiển thị ${books.length} / ${totalElements} cuốn sách`
          )}
        </span>
        {!searchTerm && totalPages > 1 && (
          <span className={styles.pageInfo}>
            Trang {currentPage} / {totalPages}
          </span>
        )}
      </div>

      {/* Books Table */}
      <div className={styles.tableContainer}>
        {books.length > 0 ? (
          <table className={styles.bookTable}>
            <thead>
              <tr>
                <th>
                  <span className={styles.headerIcon}>📖</span>
                  Thông tin sách
                </th>
                <th>
                  <span className={styles.headerIcon}>👤</span>
                  Tác giả
                </th>
                <th>
                  <span className={styles.headerIcon}>🏷️</span>
                  Thể loại
                </th>
                <th>
                  <span className={styles.headerIcon}>🏢</span>
                  Nhà xuất bản
                </th>
                <th>
                  <span className={styles.headerIcon}>📊</span>
                  Số lượng
                </th>
                <th>
                  <span className={styles.headerIcon}>⚙️</span>
                  Thao tác
                </th>
              </tr>
            </thead>
            <tbody>
              {books.map((book) => (
                <tr key={book.bookId} className={styles.bookRow}>
                  <td className={styles.bookInfoCell}>
                    <div className={styles.bookInfo}>
                      <img 
                        src={book.coverImageUrl} 
                        alt="Book Cover" 
                        className={styles.bookCover}
                        onError={(e) => {
                          e.target.src = '/placeholder-book.png';
                        }}
                      />
                      <div className={styles.bookDetails}>
                        <h4 className={styles.bookTitle}>{book.title}</h4>
                        <p className={styles.bookIsbn}>ISBN: {book.isbn}</p>
                      </div>
                    </div>
                  </td>
                  <td className={styles.authorCell}>
                    <span className={styles.author}>{book.author}</span>
                  </td>
                  <td className={styles.genreCell}>
                    <span className={styles.genreBadge}>{book.genre}</span>
                  </td>
                  <td className={styles.publisherCell}>
                    <span className={styles.publisher}>{book.publisher}</span>
                  </td>
                  <td className={styles.quantityCell}>
                    <div className={styles.quantityInfo}>
                      <div className={styles.quantityRow}>
                        <span className={styles.quantityLabel}>Tổng:</span>
                        <span className={styles.quantityValue}>{book.quantity}</span>
                      </div>
                      <div className={styles.quantityRow}>
                        <span className={styles.quantityLabel}>Có sẵn:</span>
                        <span className={styles.quantityValue}>{book.availableQuantity}</span>
                      </div>
                    </div>
                  </td>
                  <td className={styles.actionsCell}>
                    <div className={styles.actionButtons}>
                      <button 
                        className={styles.editButton} 
                        onClick={() => handleEdit(book)}
                        title="Chỉnh sửa"
                      >
                        ✏️
                      </button>
                      <button 
                        className={styles.deleteButton} 
                        onClick={() => handleDelete(book.bookId)}
                        title="Xóa"
                      >
                        🗑️
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className={styles.emptyState}>
            <div className={styles.emptyIcon}>📚</div>
            <h3 className={styles.emptyTitle}>Không tìm thấy sách nào</h3>
            <p className={styles.emptyDescription}>
              {searchTerm 
                ? `Không có sách nào phù hợp với từ khóa "${searchTerm}".`
                : "Không có sách nào phù hợp với tiêu chí tìm kiếm của bạn."
              }
            </p>
          </div>
        )}
      </div>

      {/* Pagination */}
      {!searchTerm && totalPages > 1 && (
        <div className={styles.paginationContainer}>
          <div className={styles.paginationInfo}>
            Trang {currentPage} / {totalPages} - Tổng {totalElements} cuốn sách
          </div>
          <div className={styles.pagination}>
            {renderPagination()}
          </div>
        </div>
      )}

      {showModal && (
        <BookModal
          book={currentBook}
          onClose={handleModalClose}
          onSave={handleSave}
        />
      )}
    </div>
  );
};

export default BookManagement;
