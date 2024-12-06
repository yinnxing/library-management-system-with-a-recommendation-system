package com.library.booksystem.util;

import com.library.booksystem.exception.AppException;
import com.library.booksystem.exception.ErrorCode;
import com.library.booksystem.model.specification.criteria.PaginationCriteria;
import lombok.extern.slf4j.Slf4j;
import org.apache.coyote.BadRequestException;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Slf4j
public class PageRequestBuilder extends AbstractBaseSortDirection {
    public static PageRequest build(final PaginationCriteria paginationCriteria) throws BadRequestException {
        if (paginationCriteria.getPage() == null || paginationCriteria.getPage() < 1) {
            log.warn("Page number is not valid");
            throw new AppException(ErrorCode.PAGE_NUMBER_INVALID);
        }

        paginationCriteria.setPage(paginationCriteria.getPage() - 1);

        if (paginationCriteria.getSize() == null || paginationCriteria.getSize() < 1) {
            log.warn("Page size is not valid");
            throw new AppException(ErrorCode.SIZE_INVALID);
        }

        PageRequest pageRequest = PageRequest.of(paginationCriteria.getPage(), paginationCriteria.getSize());

        if (paginationCriteria.getSortBy() != null && paginationCriteria.getSort() != null) {
            Sort.Direction direction = getDirection(paginationCriteria.getSort());

            List<String> columnsList = new ArrayList<>(Arrays.asList(paginationCriteria.getColumns()));
            if (columnsList.contains(paginationCriteria.getSortBy())) {
                return pageRequest.withSort(Sort.by(direction, paginationCriteria.getSortBy()));
            }
        }

        return pageRequest;
    }
}
